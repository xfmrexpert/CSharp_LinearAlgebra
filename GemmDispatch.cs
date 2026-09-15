using System.Runtime.CompilerServices;

namespace GemmLab;

/// <summary>
/// Owns the packing buffers for both GEMM paths and picks between them by
/// problem size.
///
/// LU needs this rather than a fixed choice. Its trailing update shrinks with
/// every block step — roughly (m-jb)*(n-jb)*nb flops, from the full matrix down
/// to nb^3 at the end — so no single answer is right for the whole
/// factorization. Calling <see cref="ParallelGemm"/> for the small tail pays
/// fork/join on work that cannot cover it; calling <see cref="Gemm"/> for the
/// first few updates leaves the machine idle during the bulk of the flops.
///
/// The dispatch is explicit rather than hidden inside <see cref="Gemm"/>
/// because the choice changes measured behaviour, and a caller benchmarking one
/// path should not silently get the other.
/// </summary>
public sealed unsafe class GemmDispatch : IDisposable
{
    /// <summary>
    /// Work (2*m*n*k flops) below which the serial path wins.
    ///
    /// A Parallel.For fork/join costs on the order of tens of microseconds, and
    /// ParallelGemm issues 2 + ceil(m/MC) of them per k-slab. At roughly
    /// 50 GFLOP/s single-core, 4e6 multiply-adds is about 160 us of work, which
    /// covers several such barriers.
    ///
    /// Derived from that argument, not measured. Sweep it.
    /// </summary>
    public const long ParallelThreshold = 4_000_000;

    private GemmScratch? _serial;
    private ParallelGemmScratch? _parallel;

    private GemmDispatch(GemmScratch serial, ParallelGemmScratch? parallel)
    {
        _serial = serial;
        _parallel = parallel;
    }

    /// <summary>Single-threaded only. Every Multiply call takes the serial path.</summary>
    public static GemmDispatch Serial<TKernel>() where TKernel : struct, IMicroKernel =>
        new(GemmScratch.For<TKernel>(), null);

    /// <summary>
    /// Multi-threaded above <see cref="ParallelThreshold"/>, serial below it.
    /// </summary>
    public static GemmDispatch Multithreaded<TKernel>(int threads = 0)
        where TKernel : struct, IMicroKernel =>
        new(GemmScratch.For<TKernel>(), ParallelGemmScratch.For<TKernel>(threads));

    /// <summary>Whether a multi-threaded path is available at all.</summary>
    public bool IsMultithreaded => _parallel is not null;

    /// <summary>
    /// C := beta*C + alpha*A*B, taking whichever path suits the problem size.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Multiply<TKernel>(
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc)
        where TKernel : struct, IMicroKernel
    {
        ObjectDisposedException.ThrowIf(_serial is null, this);

        if (_parallel is not null && (long)m * n * k >= ParallelThreshold)
        {
            ParallelGemm.Multiply<TKernel>(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, _parallel);
        }
        else
        {
            Gemm.Multiply<TKernel>(
                m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, _serial!);
        }
    }

    /// <summary>
    /// C := beta*C + alpha*A*B on the single-threaded path regardless of size.
    /// For measurements that need the serial number specifically.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void MultiplySerial<TKernel>(
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc)
        where TKernel : struct, IMicroKernel
    {
        ObjectDisposedException.ThrowIf(_serial is null, this);

        Gemm.Multiply<TKernel>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, _serial!);
    }

    public void Dispose()
    {
        _serial?.Dispose();
        _serial = null;

        _parallel?.Dispose();
        _parallel = null;

        GC.SuppressFinalize(this);
    }
}
