using System.Runtime.Intrinsics.X86;
using Tensile.Primitives;

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
    {
        _kernel = Avx512Kernel16x8.IsSupported ? Kernel.Avx512
            : Avx2Kernel8x6.IsSupported ? Kernel.Avx2
            : Kernel.Scalar;

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
    public unsafe void Multiply(
        ReadOnlyMatrixView<double> a,
        ReadOnlyMatrixView<double> b,
        MatrixView<double> c,
        double alpha = 1.0,
        double beta = 0.0)
    {
        if (a.Columns != b.Rows)
            throw new ArgumentException(
                $"Inner dimensions disagree: A is {a.Rows}x{a.Columns}, B is {b.Rows}x{b.Columns}.", nameof(b));

        if (c.Rows != a.Rows || c.Columns != b.Columns)
            throw new ArgumentException(
                $"Destination is {c.Rows}x{c.Columns}, expected {a.Rows}x{b.Columns}.", nameof(c));

        int m = a.Rows, n = b.Columns, k = a.Columns;

        lock (_gate)
        {
            GemmDispatch dispatch = Active;

            switch (_kernel)
            {
                case Kernel.Avx512:
                    dispatch.Multiply<Avx512Kernel16x8>(m, n, k, alpha, a.Pointer, a.Stride,
                        b.Pointer, b.Stride, beta, c.Pointer, c.Stride);
                    break;
                case Kernel.Avx2:
                    dispatch.Multiply<Avx2Kernel8x6>(m, n, k, alpha, a.Pointer, a.Stride,
                        b.Pointer, b.Stride, beta, c.Pointer, c.Stride);
                    break;
                default:
                    dispatch.Multiply<ScalarKernel4x4>(m, n, k, alpha, a.Pointer, a.Stride,
                        b.Pointer, b.Stride, beta, c.Pointer, c.Stride);
                    break;
            }
        }
    }

    /// <summary>
    /// Factor <paramref name="a"/> in place as P*A = L*U. The caller's storage
    /// is overwritten with the packed factors.
    /// </summary>
    /// <param name="a">The matrix to factor, overwritten.</param>
    /// <param name="blockSize">Panel width; zero selects the default.</param>
    /// <returns>The pivot array and diagnostics. The caller owns the factors.</returns>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed.</exception>
    internal unsafe LuFactorization FactorLu(MatrixView<double> a, int blockSize)
    {
        lock (_gate)
        {
            GemmDispatch dispatch = Active;
            int nb = blockSize <= 0 ? Lu.DefaultBlockSize : blockSize;

            return _kernel switch
            {
                Kernel.Avx512 => Lu.Factor<Avx512Kernel16x8>(a.Rows, a.Columns, a.Pointer, a.Stride, dispatch, nb),
                Kernel.Avx2 => Lu.Factor<Avx2Kernel8x6>(a.Rows, a.Columns, a.Pointer, a.Stride, dispatch, nb),
                _ => Lu.Factor<ScalarKernel4x4>(a.Rows, a.Columns, a.Pointer, a.Stride, dispatch, nb),
            };
        }
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
