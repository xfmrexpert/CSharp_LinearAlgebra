using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace GemmLab;

/// <summary>
/// A BLIS-style micro-kernel. Computes, for a single MR x NR tile of C:
///
///     C[0..MR, 0..NR] += alpha * Ap[MR x kc] * Bp[kc x NR]
///
/// Ap is a packed A micro-panel: for each k, MR contiguous doubles (rows).
/// Bp is a packed B micro-panel: for each k, NR contiguous doubles (columns).
/// C is column-major with column stride ldc and unit row stride.
///
/// Implementations are structs so the JIT monomorphises the driver and
/// devirtualises every call here.
/// </summary>
public interface IMicroKernel
{
    /// <summary>Rows of the C tile. Must be a multiple of the vector width.</summary>
    static abstract int Mr { get; }

    /// <summary>Columns of the C tile.</summary>
    static abstract int Nr { get; }

    /// <summary>Whether the host CPU supports this kernel's instruction set.</summary>
    static abstract bool IsSupported { get; }

    /// <summary>Short name used in reports.</summary>
    static abstract string Name { get; }

    static abstract unsafe void Execute(
        int kc, double alpha, double* ap, double* bp, double* c, int ldc);
}

/// <summary>
/// AVX-512 kernel, MR=16 NR=8.
///
/// 16 zmm accumulators (2 per column x 8 columns), plus 2 registers for the A
/// vectors and 1 for the broadcast B scalar = 19 live zmm out of 32. Per k
/// iteration: 2 vector loads, 8 broadcasts, 16 FMAs.
///
/// This is the register-pressure question in concentrated form. If RyuJIT
/// spills any of c00..c17 to the stack inside the k-loop, performance collapses
/// and the disassembly will show rsp-relative vmovupd stores in the loop body.
/// </summary>
public readonly struct Avx512Kernel16x8 : IMicroKernel
{
    public static int Mr => 16;
    public static int Nr => 8;
    public static bool IsSupported => Avx512F.IsSupported;
    public static string Name => "AVX-512 16x8";

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Execute(
        int kc, double alpha, double* ap, double* bp, double* c, int ldc)
    {
        Vector512<double> c00 = Vector512<double>.Zero, c10 = Vector512<double>.Zero;
        Vector512<double> c01 = Vector512<double>.Zero, c11 = Vector512<double>.Zero;
        Vector512<double> c02 = Vector512<double>.Zero, c12 = Vector512<double>.Zero;
        Vector512<double> c03 = Vector512<double>.Zero, c13 = Vector512<double>.Zero;
        Vector512<double> c04 = Vector512<double>.Zero, c14 = Vector512<double>.Zero;
        Vector512<double> c05 = Vector512<double>.Zero, c15 = Vector512<double>.Zero;
        Vector512<double> c06 = Vector512<double>.Zero, c16 = Vector512<double>.Zero;
        Vector512<double> c07 = Vector512<double>.Zero, c17 = Vector512<double>.Zero;

        for (int p = 0; p < kc; p++)
        {
            Vector512<double> a0 = Avx512F.LoadVector512(ap);
            Vector512<double> a1 = Avx512F.LoadVector512(ap + 8);
            ap += 16;

            Vector512<double> b;

            b = Vector512.Create(bp[0]);
            c00 = Avx512F.FusedMultiplyAdd(a0, b, c00);
            c10 = Avx512F.FusedMultiplyAdd(a1, b, c10);

            b = Vector512.Create(bp[1]);
            c01 = Avx512F.FusedMultiplyAdd(a0, b, c01);
            c11 = Avx512F.FusedMultiplyAdd(a1, b, c11);

            b = Vector512.Create(bp[2]);
            c02 = Avx512F.FusedMultiplyAdd(a0, b, c02);
            c12 = Avx512F.FusedMultiplyAdd(a1, b, c12);

            b = Vector512.Create(bp[3]);
            c03 = Avx512F.FusedMultiplyAdd(a0, b, c03);
            c13 = Avx512F.FusedMultiplyAdd(a1, b, c13);

            b = Vector512.Create(bp[4]);
            c04 = Avx512F.FusedMultiplyAdd(a0, b, c04);
            c14 = Avx512F.FusedMultiplyAdd(a1, b, c14);

            b = Vector512.Create(bp[5]);
            c05 = Avx512F.FusedMultiplyAdd(a0, b, c05);
            c15 = Avx512F.FusedMultiplyAdd(a1, b, c15);

            b = Vector512.Create(bp[6]);
            c06 = Avx512F.FusedMultiplyAdd(a0, b, c06);
            c16 = Avx512F.FusedMultiplyAdd(a1, b, c16);

            b = Vector512.Create(bp[7]);
            c07 = Avx512F.FusedMultiplyAdd(a0, b, c07);
            c17 = Avx512F.FusedMultiplyAdd(a1, b, c17);

            bp += 8;
        }

        Vector512<double> va = Vector512.Create(alpha);
        Update(c + 0 * ldc, va, c00, c10);
        Update(c + 1 * ldc, va, c01, c11);
        Update(c + 2 * ldc, va, c02, c12);
        Update(c + 3 * ldc, va, c03, c13);
        Update(c + 4 * ldc, va, c04, c14);
        Update(c + 5 * ldc, va, c05, c15);
        Update(c + 6 * ldc, va, c06, c16);
        Update(c + 7 * ldc, va, c07, c17);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Update(
        double* col, Vector512<double> alpha, Vector512<double> lo, Vector512<double> hi)
    {
        Avx512F.Store(col, Avx512F.FusedMultiplyAdd(alpha, lo, Avx512F.LoadVector512(col)));
        Avx512F.Store(col + 8, Avx512F.FusedMultiplyAdd(alpha, hi, Avx512F.LoadVector512(col + 8)));
    }
}

/// <summary>
/// AVX2 + FMA kernel, MR=8 NR=6. The classic Haswell dgemm geometry:
/// 12 ymm accumulators, 2 for A, 1 for the B broadcast = 15 of 16.
/// </summary>
public readonly struct Avx2Kernel8x6 : IMicroKernel
{
    public static int Mr => 8;
    public static int Nr => 6;
    public static bool IsSupported => Fma.IsSupported && Avx.IsSupported;
    public static string Name => "AVX2/FMA 8x6";

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Execute(
        int kc, double alpha, double* ap, double* bp, double* c, int ldc)
    {
        Vector256<double> c00 = Vector256<double>.Zero, c10 = Vector256<double>.Zero;
        Vector256<double> c01 = Vector256<double>.Zero, c11 = Vector256<double>.Zero;
        Vector256<double> c02 = Vector256<double>.Zero, c12 = Vector256<double>.Zero;
        Vector256<double> c03 = Vector256<double>.Zero, c13 = Vector256<double>.Zero;
        Vector256<double> c04 = Vector256<double>.Zero, c14 = Vector256<double>.Zero;
        Vector256<double> c05 = Vector256<double>.Zero, c15 = Vector256<double>.Zero;

        for (int p = 0; p < kc; p++)
        {
            Vector256<double> a0 = Avx.LoadVector256(ap);
            Vector256<double> a1 = Avx.LoadVector256(ap + 4);
            ap += 8;

            Vector256<double> b;

            b = Vector256.Create(bp[0]);
            c00 = Fma.MultiplyAdd(a0, b, c00);
            c10 = Fma.MultiplyAdd(a1, b, c10);

            b = Vector256.Create(bp[1]);
            c01 = Fma.MultiplyAdd(a0, b, c01);
            c11 = Fma.MultiplyAdd(a1, b, c11);

            b = Vector256.Create(bp[2]);
            c02 = Fma.MultiplyAdd(a0, b, c02);
            c12 = Fma.MultiplyAdd(a1, b, c12);

            b = Vector256.Create(bp[3]);
            c03 = Fma.MultiplyAdd(a0, b, c03);
            c13 = Fma.MultiplyAdd(a1, b, c13);

            b = Vector256.Create(bp[4]);
            c04 = Fma.MultiplyAdd(a0, b, c04);
            c14 = Fma.MultiplyAdd(a1, b, c14);

            b = Vector256.Create(bp[5]);
            c05 = Fma.MultiplyAdd(a0, b, c05);
            c15 = Fma.MultiplyAdd(a1, b, c15);

            bp += 6;
        }

        Vector256<double> va = Vector256.Create(alpha);
        Update(c + 0 * ldc, va, c00, c10);
        Update(c + 1 * ldc, va, c01, c11);
        Update(c + 2 * ldc, va, c02, c12);
        Update(c + 3 * ldc, va, c03, c13);
        Update(c + 4 * ldc, va, c04, c14);
        Update(c + 5 * ldc, va, c05, c15);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Update(
        double* col, Vector256<double> alpha, Vector256<double> lo, Vector256<double> hi)
    {
        Avx.Store(col, Fma.MultiplyAdd(alpha, lo, Avx.LoadVector256(col)));
        Avx.Store(col + 4, Fma.MultiplyAdd(alpha, hi, Avx.LoadVector256(col + 4)));
    }
}

/// <summary>
/// Portable scalar kernel, MR=4 NR=4. Correctness fallback and a sanity
/// baseline showing what the packing and blocking alone buy you.
/// </summary>
public readonly struct ScalarKernel4x4 : IMicroKernel
{
    public static int Mr => 4;
    public static int Nr => 4;
    public static bool IsSupported => true;
    public static string Name => "scalar 4x4";

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe void Execute(
        int kc, double alpha, double* ap, double* bp, double* c, int ldc)
    {
        double c00 = 0, c10 = 0, c20 = 0, c30 = 0;
        double c01 = 0, c11 = 0, c21 = 0, c31 = 0;
        double c02 = 0, c12 = 0, c22 = 0, c32 = 0;
        double c03 = 0, c13 = 0, c23 = 0, c33 = 0;

        for (int p = 0; p < kc; p++)
        {
            double a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
            ap += 4;

            double b0 = bp[0], b1 = bp[1], b2 = bp[2], b3 = bp[3];
            bp += 4;

            c00 += a0 * b0; c10 += a1 * b0; c20 += a2 * b0; c30 += a3 * b0;
            c01 += a0 * b1; c11 += a1 * b1; c21 += a2 * b1; c31 += a3 * b1;
            c02 += a0 * b2; c12 += a1 * b2; c22 += a2 * b2; c32 += a3 * b2;
            c03 += a0 * b3; c13 += a1 * b3; c23 += a2 * b3; c33 += a3 * b3;
        }

        double* col = c;
        col[0] += alpha * c00; col[1] += alpha * c10; col[2] += alpha * c20; col[3] += alpha * c30;
        col += ldc;
        col[0] += alpha * c01; col[1] += alpha * c11; col[2] += alpha * c21; col[3] += alpha * c31;
        col += ldc;
        col[0] += alpha * c02; col[1] += alpha * c12; col[2] += alpha * c22; col[3] += alpha * c32;
        col += ldc;
        col[0] += alpha * c03; col[1] += alpha * c13; col[2] += alpha * c23; col[3] += alpha * c33;
    }
}
