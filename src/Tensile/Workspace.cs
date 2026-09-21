using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// The reusable buffers and kernel choice behind every operation.
///
/// Two things have to be decided before a GEMM can run: which micro-kernel the
/// host CPU supports, and where the packed panels live. Both are expensive to
/// redo per call — the packed-B buffer alone is sized to occupy a good fraction
/// of L3 — so they are bundled here and kept alive across calls.
///
/// Methods are internally locked, which makes an instance safe to share and
/// costs an uncontended lock against work that is at minimum O(n^2). A caller
/// that wants genuine concurrency should give each thread its own workspace;
/// note that the multi-threaded GEMM already uses every core from within a
/// single call, so running several at once usually buys nothing.
///
/// <see cref="Shared"/> exists so the convenience API has something to use. It
/// is the right default for ordinary code and the wrong one for a benchmark,
/// where the lock and the shared buffers are measurement noise.
///
/// The workspace itself remains disposable because its packing buffers are
/// native memory owned by the kernel layer; the allocator phase of the
/// security migration revisits that. The matrices it operates on are not
/// disposable and never were a lifetime risk through this type, since it holds
/// no reference to them between calls.
/// </summary>
public sealed class Workspace : IDisposable
{
    private enum Kernel { Avx512, Avx2, Scalar }

    private static readonly Lazy<Workspace> LazyShared = new(() => new Workspace(multithreaded: true));

    private readonly Lock _gate = new();
    private readonly Kernel _kernel;
    private GemmDispatch? _dispatch;

    /// <summary>Create a workspace using the widest micro-kernel this CPU supports.</summary>
    /// <param name="multithreaded">
    /// Whether large products may use several threads. Small ones fall back to
    /// the serial path regardless, since fork/join costs more than they save.
    /// </param>
    public Workspace(bool multithreaded = true)
        : this(Avx512Kernel16x8.IsSupported ? Kernel.Avx512 : Avx2Kernel8x6.IsSupported ? Kernel.Avx2 : Kernel.Scalar,
            multithreaded)
    {
    }

    private Workspace(Kernel kernel, bool multithreaded)
    {
        _kernel = kernel;
        IsMultithreaded = multithreaded;

        _dispatch = (_kernel, multithreaded) switch
        {
            (Kernel.Avx512, true) => GemmDispatch.Multithreaded<Avx512Kernel16x8>(),
            (Kernel.Avx512, false) => GemmDispatch.Serial<Avx512Kernel16x8>(),
            (Kernel.Avx2, true) => GemmDispatch.Multithreaded<Avx2Kernel8x6>(),
            (Kernel.Avx2, false) => GemmDispatch.Serial<Avx2Kernel8x6>(),
            (_, true) => GemmDispatch.Multithreaded<ScalarKernel4x4>(),
            (_, false) => GemmDispatch.Serial<ScalarKernel4x4>(),
        };
    }

    /// <summary>
    /// A workspace on a named micro-kernel, for the kernel-generic test
    /// contracts, which run the public operations once per kernel the host
    /// supports. Callers check <c>TKernel.IsSupported</c> first; a workspace on
    /// an unsupported kernel faults on first use.
    /// </summary>
    internal static Workspace ForKernel<TKernel>(bool multithreaded) where TKernel : struct, IMicroKernel
    {
        Kernel kernel = typeof(TKernel) == typeof(Avx512Kernel16x8) ? Kernel.Avx512
            : typeof(TKernel) == typeof(Avx2Kernel8x6) ? Kernel.Avx2
            : typeof(TKernel) == typeof(ScalarKernel4x4) ? Kernel.Scalar
            : throw new ArgumentException($"Unknown micro-kernel {typeof(TKernel).Name}.");

        return new Workspace(kernel, multithreaded);
    }

    /// <summary>
    /// A process-wide workspace, used by any operation that is not given one.
    /// Never disposed; its buffers live for the lifetime of the process.
    /// </summary>
    public static Workspace Shared => LazyShared.Value;

    /// <summary>Name of the selected micro-kernel, for diagnostics.</summary>
    public string KernelName => _kernel switch
    {
        Kernel.Avx512 => Avx512Kernel16x8.Name,
        Kernel.Avx2 => Avx2Kernel8x6.Name,
        _ => ScalarKernel4x4.Name,
    };

    /// <summary>Whether large products may use several threads.</summary>
    public bool IsMultithreaded { get; }

    /// <summary>
    /// C := beta*C + alpha*A*B, with the shapes taken from the views.
    /// </summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="c">Destination, m x n. Overwritten.</param>
    /// <param name="alpha">Scalar on the product.</param>
    /// <param name="beta">Scalar on the existing contents of C. Zero overwrites rather than scales, so a C full of NaN still yields a finite result.</param>
    /// <exception cref="ArgumentException">The shapes are not conformable.</exception>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed.</exception>
    public void Multiply(
        ReadOnlyMatrixView<double> a,
        ReadOnlyMatrixView<double> b,
        MatrixView<double> c,
        double alpha = 1.0,
        double beta = 0.0)
    {
        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Inner dimensions disagree: A is {a.Rows}x{a.Columns}, B is {b.Rows}x{b.Columns}.", nameof(b));
        }

        if (c.Rows != a.Rows || c.Columns != b.Columns)
        {
            throw new ArgumentException(
                $"Destination is {c.Rows}x{c.Columns}, expected {a.Rows}x{b.Columns}.", nameof(c));
        }

        lock (_gate)
        {
            GemmDispatch dispatch = Active;

            switch (_kernel)
            {
                case Kernel.Avx512:
                    KernelEntry.Multiply<Avx512Kernel16x8>(dispatch, a.ToOperand(), b.ToOperand(), c.ToTarget(), alpha, beta);
                    break;
                case Kernel.Avx2:
                    KernelEntry.Multiply<Avx2Kernel8x6>(dispatch, a.ToOperand(), b.ToOperand(), c.ToTarget(), alpha, beta);
                    break;
                default:
                    KernelEntry.Multiply<ScalarKernel4x4>(dispatch, a.ToOperand(), b.ToOperand(), c.ToTarget(), alpha, beta);
                    break;
            }
        }
    }

    /// <summary>
    /// Factor <paramref name="a"/> in place as P*A = L*U, without copying.
    /// The matrix is overwritten with the packed factors and the returned
    /// decomposition shares its storage, so <paramref name="a"/> must not be
    /// written afterwards while the decomposition is in use.
    ///
    /// This is the zero-copy path; <see cref="MatrixOperations.FactorLu"/> is
    /// the copying one and the right default. Taking a <see cref="Matrix{T}"/>
    /// rather than a view is deliberate: the decomposition has to keep the
    /// factors alive for as long as it exists, and a borrowed view cannot
    /// promise that, but a matrix -- a garbage-collected object -- can.
    /// </summary>
    /// <param name="a">The square or rectangular matrix to factor. Overwritten.</param>
    /// <param name="blockSize">Panel width; zero selects the default. The optimum shifts with size.</param>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed.</exception>
    public LuDecomposition FactorLu(Matrix<double> a, int blockSize = 0) =>
        FactorLu(a, blockSize, timings: null);

    /// <summary>
    /// The same factorization with per-phase timing collected into
    /// <paramref name="timings"/>. Internal because it exists for the
    /// diagnostics tool and the tests: the phase split is a measurement input,
    /// not part of the library's contract, and a null collector is exactly the
    /// shipped path.
    /// </summary>
    internal LuDecomposition FactorLu(Matrix<double> a, int blockSize, LuPhaseTimings? timings)
    {
        ArgumentNullException.ThrowIfNull(a);

        // Captured before the factorization overwrites the matrix, so that
        // condition estimation later cannot be given the wrong norm.
        double oneNorm = a.OneNorm();

        LuFactorization factorization;

        lock (_gate)
        {
            GemmDispatch dispatch = Active;
            int nb = blockSize <= 0 ? Lu.DefaultBlockSizeFor(a.Rows, a.Columns) : blockSize;
            Target target = a.View.ToTarget();

            factorization = _kernel switch
            {
                Kernel.Avx512 => KernelEntry.FactorLu<Avx512Kernel16x8>(dispatch, target, nb, timings),
                Kernel.Avx2 => KernelEntry.FactorLu<Avx2Kernel8x6>(dispatch, target, nb, timings),
                _ => KernelEntry.FactorLu<ScalarKernel4x4>(dispatch, target, nb, timings),
            };
        }

        return new LuDecomposition(a, factorization, oneNorm);
    }

    private GemmDispatch Active
    {
        get
        {
            ObjectDisposedException.ThrowIf(_dispatch is null, this);
            return _dispatch!;
        }
    }

    /// <summary>Release the packing buffers. Safe to call more than once.</summary>
    public void Dispose()
    {
        lock (_gate)
        {
            _dispatch?.Dispose();
            _dispatch = null;
        }
    }
}
