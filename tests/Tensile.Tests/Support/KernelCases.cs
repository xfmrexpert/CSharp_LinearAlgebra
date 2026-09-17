using Tensile.Kernels;

namespace Tensile.Tests;

/// <summary>
/// Names a micro-kernel for a test contract.
///
/// The kernel types are internal to <c>Tensile.Kernels</c> -- that is the
/// visibility half of invariant I4 -- and xunit requires a test class to be
/// public, so a public contract cannot be generic over a kernel type directly.
/// It is generic over one of these instead, a public marker that says which
/// kernel it means, and reaches the kernel through <see cref="KernelDriver"/>.
/// </summary>
public interface IKernelCase
{
    /// <summary>Which micro-kernel this case stands for.</summary>
    static abstract KernelId Id { get; }
}

/// <summary>The micro-kernels a contract can run against.</summary>
public enum KernelId
{
    /// <summary>The portable 4x4 kernel, supported everywhere.</summary>
    Scalar,

    /// <summary>The AVX2/FMA 8x6 kernel.</summary>
    Avx2,

    /// <summary>The AVX-512 16x8 kernel.</summary>
    Avx512,
}

/// <summary>Case for <c>ScalarKernel4x4</c>.</summary>
public readonly struct ScalarCase : IKernelCase
{
    /// <inheritdoc/>
    public static KernelId Id => KernelId.Scalar;
}

/// <summary>Case for <c>Avx2Kernel8x6</c>.</summary>
public readonly struct Avx2Case : IKernelCase
{
    /// <inheritdoc/>
    public static KernelId Id => KernelId.Avx2;
}

/// <summary>Case for <c>Avx512Kernel16x8</c>.</summary>
public readonly struct Avx512Case : IKernelCase
{
    /// <inheritdoc/>
    public static KernelId Id => KernelId.Avx512;
}

/// <summary>
/// Everything a contract does with a kernel, with the kernel type bound inside.
/// One instance per kernel; the contracts obtain theirs from <see cref="For"/>.
/// </summary>
internal abstract unsafe class KernelDriver
{
    public abstract bool IsSupported { get; }
    public abstract string Name { get; }

    public abstract GemmDispatch Serial();
    public abstract GemmDispatch Multithreaded();

    public abstract void Multiply(
        GemmDispatch gemm, int m, int n, int k,
        double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc);

    public abstract void MultiplySerial(
        GemmDispatch gemm, int m, int n, int k,
        double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc);

    /// <summary>The threaded path directly, with a fresh scratch, bypassing the dispatch threshold.</summary>
    public abstract void MultiplyParallel(
        int m, int n, int k,
        double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc);

    public abstract LuFactorization FactorLu(int m, int n, double* a, int lda, GemmDispatch gemm, int blockSize);

    /// <summary>A public-layer workspace pinned to this kernel.</summary>
    public abstract Workspace Workspace(bool multithreaded);

    public static KernelDriver For<TCase>() where TCase : struct, IKernelCase => TCase.Id switch
    {
        KernelId.Scalar => Bound<ScalarKernel4x4>.Instance,
        KernelId.Avx2 => Bound<Avx2Kernel8x6>.Instance,
        KernelId.Avx512 => Bound<Avx512Kernel16x8>.Instance,
        _ => throw new ArgumentOutOfRangeException(nameof(TCase), $"Unknown kernel case {TCase.Id}."),
    };

    private sealed class Bound<TKernel> : KernelDriver where TKernel : struct, IMicroKernel
    {
        public static readonly Bound<TKernel> Instance = new();

        public override bool IsSupported => TKernel.IsSupported;
        public override string Name => TKernel.Name;

        public override GemmDispatch Serial() => GemmDispatch.Serial<TKernel>();
        public override GemmDispatch Multithreaded() => GemmDispatch.Multithreaded<TKernel>();

        public override void Multiply(
            GemmDispatch gemm, int m, int n, int k,
            double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc) =>
            gemm.Multiply<TKernel>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

        public override void MultiplySerial(
            GemmDispatch gemm, int m, int n, int k,
            double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc) =>
            gemm.MultiplySerial<TKernel>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);

        public override void MultiplyParallel(
            int m, int n, int k,
            double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc)
        {
            using var scratch = ParallelGemmScratch.For<TKernel>();
            ParallelGemm.Multiply<TKernel>(m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, scratch);
        }

        public override LuFactorization FactorLu(int m, int n, double* a, int lda, GemmDispatch gemm, int blockSize) =>
            Lu.Factor<TKernel>(m, n, a, lda, gemm, blockSize);

        public override Workspace Workspace(bool multithreaded) => Tensile.Workspace.ForKernel<TKernel>(multithreaded);
    }
}
