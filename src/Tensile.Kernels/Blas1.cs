using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tensile.Kernels;

/// <summary>
/// Level-1 primitives. These exist because the LU panel factorization is
/// level-2 work that GEMM cannot absorb: for an m x n factorization with block
/// size nb it accounts for roughly m*n*nb flops, about 9% of the total at
/// n=1024, nb=64. Left scalar, it runs ~25x slower than the GEMM around it and
/// dominates the runtime. Vectorised, it disappears into the noise.
///
/// Vector&lt;double&gt; rather than explicit intrinsics: this is bandwidth-bound
/// streaming work, not FMA-issue-bound, so the portable width is enough and it
/// keeps the file architecture-neutral.
/// </summary>
internal static unsafe class Blas1
{
    /// <summary>
    /// Index of the element with largest magnitude, ties resolving to the
    /// lowest index. Returns 0 for n &lt;= 0.
    ///
    /// The tie rule is not cosmetic: it is what makes the pivot sequence
    /// reproducible, and it matches LAPACK's idamax, so a factorization can be
    /// compared against one produced there.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int IndexOfMaxAbs(int n, double* x)
    {
        if (n <= 0) return 0;

        int width = Vector<double>.Count;
        int best = 0;
        double bestValue = Math.Abs(x[0]);
        int i = 1;

        // Below two vector widths the reduction costs more than it saves.
        if (Vector.IsHardwareAccelerated && n >= 2 * width)
        {
            Span<long> lanes = stackalloc long[width];
            for (int lane = 0; lane < width; lane++) lanes[lane] = lane;

            var indices = new Vector<long>(lanes);
            var step = new Vector<long>((long)width);

            var bestVector = Vector.Abs(new Vector<double>(new ReadOnlySpan<double>(x, width)));
            var bestIndices = indices;

            // Strict greater-than keeps the earliest index within each lane;
            // the reduction below breaks cross-lane ties the same way. Together
            // that reproduces the scalar scan exactly, NaNs included: every
            // comparison against a NaN is false in both forms.
            for (i = width; i <= n - width; i += width)
            {
                indices += step;

                var value = Vector.Abs(new Vector<double>(new ReadOnlySpan<double>(x + i, width)));
                var greater = Vector.GreaterThan(value, bestVector);

                bestVector = Vector.ConditionalSelect(greater, value, bestVector);
                bestIndices = Vector.ConditionalSelect(
                    Vector.AsVectorInt64(greater), indices, bestIndices);
            }

            bestValue = bestVector[0];
            best = (int)bestIndices[0];

            for (int lane = 1; lane < width; lane++)
            {
                double value = bestVector[lane];
                int index = (int)bestIndices[lane];

                if (value > bestValue || (value == bestValue && index < best))
                {
                    bestValue = value;
                    best = index;
                }
            }
        }

        for (; i < n; i++)
        {
            double value = Math.Abs(x[i]);
            if (value > bestValue)
            {
                bestValue = value;
                best = i;
            }
        }

        return best;
    }

    /// <summary>
    /// Sum of x[i]*y[i]. Four accumulators because the transposed triangular
    /// solves call this with short vectors — average nb/2, around 32 elements —
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

    /// <summary>Largest magnitude in a strided row. Used for norm estimates.</summary>
    public static double MaxAbsStrided(int n, double* x, int stride)
    {
        double best = 0.0;
        for (int i = 0; i < n; i++) best = Math.Max(best, Math.Abs(x[(nint)i * stride]));
        return best;
    }
}
