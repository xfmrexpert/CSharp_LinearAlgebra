namespace Tensile.Primitives;

/// <summary>
/// Unblocked reference implementation and accuracy checks. Slow on purpose:
/// this is the oracle, not a competitor.
/// </summary>
public static unsafe class Reference
{
    /// <summary>C := beta*C + alpha*A*B, column-major, textbook loop order.</summary>
    public static void Multiply(
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                double sum = 0.0;
                for (int p = 0; p < k; p++)
                    sum += a[(nint)p * lda + i] * b[(nint)j * ldb + p];

                double* target = c + (nint)j * ldc + i;
                *target = beta * *target + alpha * sum;
            }
        }
    }

    /// <summary>
    /// Relative Frobenius-norm difference, ||X - Y||_F / ||Y||_F.
    /// For a correct blocked GEMM this should sit at a few units of machine
    /// epsilon times sqrt(k) — reordered summation, not a different answer.
    /// </summary>
    public static double RelativeResidual(
        int m, int n, double* x, int ldx, double* y, int ldy)
    {
        double diff = 0.0;
        double norm = 0.0;

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                double yv = y[(nint)j * ldy + i];
                double d = x[(nint)j * ldx + i] - yv;
                diff += d * d;
                norm += yv * yv;
            }
        }

        return norm == 0.0 ? Math.Sqrt(diff) : Math.Sqrt(diff) / Math.Sqrt(norm);
    }
}
