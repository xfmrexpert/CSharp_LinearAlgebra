using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tensile.Kernels;

/// <summary>
/// Vectorised updates over a single contiguous column.
///
/// These are the streaming counterpart to <see cref="Gemm"/>: that packs its
/// operands into micro-panels because O(n^3) of work amortises the packing,
/// while everything here is O(n) per call and reads its operand exactly once,
/// so packing could never pay for itself. That distinction -- packed or
/// streamed -- is the axis these files are organised on, not the arity of the
/// operands.
///
/// They exist as shared code because the LU panel factorization and the
/// triangular solves are the two places GEMM cannot absorb the work: for an
/// m x n factorization with block size nb the panel accounts for roughly
/// m*n*nb flops, about 9% of the total at n=1024, nb=64. Left scalar, it runs
/// ~25x slower than the GEMM around it and dominates the runtime. Vectorised,
/// it disappears into the noise.
///
/// Vector&lt;double&gt; rather than explicit intrinsics: this is bandwidth-bound
/// streaming work, not FMA-issue-bound, so the portable width is enough and it
/// keeps the file architecture-neutral.
/// </summary>
internal static unsafe class ColumnOps
{
    /// <summary>
    /// Sum of x[i]*y[i]. Four accumulators because the transposed triangular
    /// solves call this with short vectors -- average nb/2, around 32 elements --
    /// where a single chain is FMA-latency bound rather than throughput bound.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static double Dot(int n, double* x, double* y)
    {
        int i = 0;
        int width = Vector<double>.Count;
        double total = 0.0;

        if (Vector.IsHardwareAccelerated && n >= width)
        {
            var a0 = Vector<double>.Zero;
            var a1 = Vector<double>.Zero;
            var a2 = Vector<double>.Zero;
            var a3 = Vector<double>.Zero;

            int stride = 4 * width;
            for (; i <= n - stride; i += stride)
            {
                a0 += Load(x + i) * Load(y + i);
                a1 += Load(x + i + width) * Load(y + i + width);
                a2 += Load(x + i + 2 * width) * Load(y + i + 2 * width);
                a3 += Load(x + i + 3 * width) * Load(y + i + 3 * width);
            }

            for (; i <= n - width; i += width) a0 += Load(x + i) * Load(y + i);

            total = Vector.Sum((a0 + a1) + (a2 + a3));
        }

        for (; i < n; i++) total += x[i] * y[i];

        return total;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<double> Load(double* p) =>
        new(new ReadOnlySpan<double>(p, Vector<double>.Count));

    /// <summary>x := alpha * x</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Scale(int n, double alpha, double* x)
    {
        int i = 0;
        int width = Vector<double>.Count;

        if (Vector.IsHardwareAccelerated && n >= width)
        {
            var va = new Vector<double>(alpha);
            for (; i <= n - width; i += width)
                (new Vector<double>(new ReadOnlySpan<double>(x + i, width)) * va)
                    .CopyTo(new Span<double>(x + i, width));
        }

        for (; i < n; i++) x[i] *= alpha;
    }

    /// <summary>y := y + alpha * x</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Axpy(int n, double alpha, double* x, double* y)
    {
        int i = 0;
        int width = Vector<double>.Count;

        if (Vector.IsHardwareAccelerated && n >= width)
        {
            var va = new Vector<double>(alpha);
            for (; i <= n - width; i += width)
            {
                var vx = new Vector<double>(new ReadOnlySpan<double>(x + i, width));
                var vy = new Vector<double>(new ReadOnlySpan<double>(y + i, width));
                (vy + va * vx).CopyTo(new Span<double>(y + i, width));
            }
        }

        for (; i < n; i++) y[i] += alpha * x[i];
    }
}
