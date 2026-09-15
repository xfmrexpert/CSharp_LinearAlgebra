using System.Numerics;
using System.Runtime.CompilerServices;

namespace GemmLab;

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
    /// <summary>Index of the element with largest magnitude. Returns 0 for n &lt;= 0.</summary>
    public static int IndexOfMaxAbs(int n, double* x)
    {
        if (n <= 0) return 0;

        int best = 0;
        double bestValue = Math.Abs(x[0]);

        for (int i = 1; i < n; i++)
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
