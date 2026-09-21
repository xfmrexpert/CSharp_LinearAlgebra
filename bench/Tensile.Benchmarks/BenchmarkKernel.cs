using Tensile.Kernels;

namespace Tensile.Benchmarks;

/// <summary>
/// Binds the widest micro-kernel this CPU supports, once, so that every
/// benchmark does not repeat the same three-way if-chain.
///
/// The kernel types are generic parameters, not values, so the choice cannot be
/// stored in a field and handed around; each method here closes over it at the
/// call site instead. The cost is a predictable branch outside the measured
/// loop, which is noise against an O(n^3) product.
/// </summary>
internal static unsafe class BenchmarkKernel
{
    private static bool HasAvx512 => Avx512Kernel16x8.IsSupported;
    private static bool HasAvx2 => Avx2Kernel8x6.IsSupported;

    /// <summary>Name of the kernel every benchmark in this project will use.</summary>
    public static string Name =>
        HasAvx512 ? Avx512Kernel16x8.Name : HasAvx2 ? Avx2Kernel8x6.Name : ScalarKernel4x4.Name;

    /// <summary>A dispatch that always takes the serial path.</summary>
    public static GemmDispatch Serial() =>
        HasAvx512 ? GemmDispatch.Serial<Avx512Kernel16x8>()
        : HasAvx2 ? GemmDispatch.Serial<Avx2Kernel8x6>()
        : GemmDispatch.Serial<ScalarKernel4x4>();

    /// <summary>A dispatch that may thread, capped at <paramref name="threads"/> (zero means every core).</summary>
    public static GemmDispatch Multithreaded(int threads = 0) =>
        HasAvx512 ? GemmDispatch.Multithreaded<Avx512Kernel16x8>(threads)
        : HasAvx2 ? GemmDispatch.Multithreaded<Avx2Kernel8x6>(threads)
        : GemmDispatch.Multithreaded<ScalarKernel4x4>(threads);

    /// <summary>A serial-only dispatch over packing buffers the caller built.</summary>
    public static GemmDispatch SerialWith(GemmScratch scratch) => GemmDispatch.SerialWith(scratch);

    /// <summary>Serial packing buffers with explicit cache-blocking parameters.</summary>
    public static GemmScratch Scratch(int mc, int kc, int nc) =>
        HasAvx512 ? GemmScratch.For<Avx512Kernel16x8>(mc, kc, nc)
        : HasAvx2 ? GemmScratch.For<Avx2Kernel8x6>(mc, kc, nc)
        : GemmScratch.For<ScalarKernel4x4>(mc, kc, nc);

    /// <summary>Threaded packing buffers, capped at <paramref name="threads"/>.</summary>
    public static ParallelGemmScratch ParallelScratch(int threads = 0) =>
        HasAvx512 ? ParallelGemmScratch.For<Avx512Kernel16x8>(threads)
        : HasAvx2 ? ParallelGemmScratch.For<Avx2Kernel8x6>(threads)
        : ParallelGemmScratch.For<ScalarKernel4x4>(threads);

    /// <summary>C := A*B through the dispatch, which picks serial or threaded by size.</summary>
    public static void Multiply(
        GemmDispatch dispatch, int m, int n, int k,
        double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        if (HasAvx512) dispatch.Multiply<Avx512Kernel16x8>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
        else if (HasAvx2) dispatch.Multiply<Avx2Kernel8x6>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
        else dispatch.Multiply<ScalarKernel4x4>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }

    /// <summary>C := A*B on the serial path regardless of size.</summary>
    public static void MultiplySerial(
        GemmDispatch dispatch, int m, int n, int k,
        double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        if (HasAvx512) dispatch.MultiplySerial<Avx512Kernel16x8>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
        else if (HasAvx2) dispatch.MultiplySerial<Avx2Kernel8x6>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
        else dispatch.MultiplySerial<ScalarKernel4x4>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }

    /// <summary>C := A*B on the serial driver with caller-chosen block sizes.</summary>
    public static void Gemm(
        GemmScratch scratch, int m, int n, int k,
        double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        if (HasAvx512) Kernels.Gemm.Multiply<Avx512Kernel16x8>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc, scratch);
        else if (HasAvx2) Kernels.Gemm.Multiply<Avx2Kernel8x6>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc, scratch);
        else Kernels.Gemm.Multiply<ScalarKernel4x4>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc, scratch);
    }

    /// <summary>C := A*B on the threaded driver, bypassing any size threshold.</summary>
    public static void ParallelGemm(
        ParallelGemmScratch scratch, int m, int n, int k,
        double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        if (HasAvx512) Kernels.ParallelGemm.Multiply<Avx512Kernel16x8>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc, scratch);
        else if (HasAvx2) Kernels.ParallelGemm.Multiply<Avx2Kernel8x6>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc, scratch);
        else Kernels.ParallelGemm.Multiply<ScalarKernel4x4>(m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc, scratch);
    }
}
