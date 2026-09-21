namespace Tensile.Kernels;

/// <summary>
/// Exact matrix norms over pointer-and-stride operands. The estimated 1-norm
/// lives in the public assembly as <c>Tensile.NormEstimate</c>: it is
/// bookkeeping around products this layer supplies, and needs no pointer.
/// </summary>
internal static unsafe class Norms
{
    /// <summary>||A||_1, the largest absolute column sum. O(m*n).</summary>
    public static double One(int m, int n, double* a, int lda)
    {
        double best = 0.0;

        for (int j = 0; j < n; j++)
        {
            double* column = a + (nint)j * lda;

            double sum = 0.0;
            for (int i = 0; i < m; i++) sum += Math.Abs(column[i]);

            best = Math.Max(best, sum);
        }

        return best;
    }

    /// <summary>
    /// ||A||_F, the square root of the sum of squares.
    ///
    /// Accumulated directly rather than by the scaled two-pass method LAPACK's
    /// dlassq uses, so it overflows for entries near the top of the range and
    /// underflows to zero for entries near the bottom. Fine for residual
    /// checks, which is what it is for here.
    /// </summary>
    public static double Frobenius(int m, int n, double* a, int lda)
    {
        double total = 0.0;

        for (int j = 0; j < n; j++)
        {
            double* column = a + (nint)j * lda;
            for (int i = 0; i < m; i++) total += column[i] * column[i];
        }

        return Math.Sqrt(total);
    }

    /// <summary>||A||_inf, the largest absolute row sum. O(m*n).</summary>
    public static double Infinity(int m, int n, double* a, int lda)
    {
        if (m == 0 || n == 0) return 0.0;

        var sums = new double[m];

        for (int j = 0; j < n; j++)
        {
            double* column = a + (nint)j * lda;
            for (int i = 0; i < m; i++) sums[i] += Math.Abs(column[i]);
        }

        double best = 0.0;
        for (int i = 0; i < m; i++) best = Math.Max(best, sums[i]);

        return best;
    }
}
