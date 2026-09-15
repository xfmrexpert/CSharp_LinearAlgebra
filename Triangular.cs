using System.Numerics;
using System.Runtime.CompilerServices;

namespace GemmLab;

/// <summary>
/// Triangular solves, column-oriented so the right-hand side is walked
/// contiguously.
///
/// Four right-hand sides are processed together. Done one column at a time the
/// inner axpy has average length nb/2 -- around 32 elements -- which is too
/// short to amortise its own setup, and the triangular factor is re-read for
/// every column. Blocking by four shares each read of L across four updates and
/// gives the scheduler independent chains to interleave.
///
/// Still unblocked in the triangular dimension. In LU the factor is only
/// nb x nb and fits L1, so recursion (which would push the bulk into GEMM)
/// is not yet worth the complexity.
/// </summary>
internal static unsafe class Triangular
{
    /// <summary>
    /// Solve L * X = B in place, L being m x m unit lower triangular (the
    /// diagonal is implicit, so LU's packed storage works directly).
    /// </summary>
    public static void SolveLowerUnit(int m, int n, double* l, int ldl, double* b, int ldb)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4) SolveLowerUnit4(m, l, ldl, b + (nint)j * ldb, ldb);
        for (; j < n; j++) SolveLowerUnit1(m, l, ldl, b + (nint)j * ldb);
    }

    /// <summary>
    /// Solve U * X = B in place, U being m x m upper triangular with an
    /// explicit diagonal.
    /// </summary>
    public static void SolveUpper(int m, int n, double* u, int ldu, double* b, int ldb)
    {
        int j = 0;
        for (; j + 4 <= n; j += 4) SolveUpper4(m, u, ldu, b + (nint)j * ldb, ldb);
        for (; j < n; j++) SolveUpper1(m, u, ldu, b + (nint)j * ldb);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SolveLowerUnit4(int m, double* l, int ldl, double* b, int ldb)
    {
        double* c0 = b, c1 = b + ldb, c2 = b + 2 * ldb, c3 = b + 3 * ldb;

        for (int p = 0; p < m - 1; p++)
        {
            double v0 = c0[p], v1 = c1[p], v2 = c2[p], v3 = c3[p];
            if (v0 == 0.0 && v1 == 0.0 && v2 == 0.0 && v3 == 0.0) continue;

            UpdateFour(m - p - 1, l + (nint)p * ldl + p + 1,
                c0 + p + 1, c1 + p + 1, c2 + p + 1, c3 + p + 1, v0, v1, v2, v3);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SolveUpper4(int m, double* u, int ldu, double* b, int ldb)
    {
        double* c0 = b, c1 = b + ldb, c2 = b + 2 * ldb, c3 = b + 3 * ldb;

        for (int p = m - 1; p >= 0; p--)
        {
            double diagonal = u[(nint)p * ldu + p];
            c0[p] /= diagonal; c1[p] /= diagonal; c2[p] /= diagonal; c3[p] /= diagonal;

            double v0 = c0[p], v1 = c1[p], v2 = c2[p], v3 = c3[p];
            if (p == 0 || (v0 == 0.0 && v1 == 0.0 && v2 == 0.0 && v3 == 0.0)) continue;

            UpdateFour(p, u + (nint)p * ldu, c0, c1, c2, c3, v0, v1, v2, v3);
        }
    }

    /// <summary>c_i := c_i - v_i * x, for four independent right-hand sides.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UpdateFour(
        int n, double* x,
        double* c0, double* c1, double* c2, double* c3,
        double v0, double v1, double v2, double v3)
    {
        int i = 0;
        int width = Vector<double>.Count;

        if (Vector.IsHardwareAccelerated && n >= width)
        {
            var a0 = new Vector<double>(-v0);
            var a1 = new Vector<double>(-v1);
            var a2 = new Vector<double>(-v2);
            var a3 = new Vector<double>(-v3);

            for (; i <= n - width; i += width)
            {
                var xv = new Vector<double>(new ReadOnlySpan<double>(x + i, width));

                (new Vector<double>(new ReadOnlySpan<double>(c0 + i, width)) + a0 * xv)
                    .CopyTo(new Span<double>(c0 + i, width));
                (new Vector<double>(new ReadOnlySpan<double>(c1 + i, width)) + a1 * xv)
                    .CopyTo(new Span<double>(c1 + i, width));
                (new Vector<double>(new ReadOnlySpan<double>(c2 + i, width)) + a2 * xv)
                    .CopyTo(new Span<double>(c2 + i, width));
                (new Vector<double>(new ReadOnlySpan<double>(c3 + i, width)) + a3 * xv)
                    .CopyTo(new Span<double>(c3 + i, width));
            }
        }

        for (; i < n; i++)
        {
            double value = x[i];
            c0[i] -= v0 * value;
            c1[i] -= v1 * value;
            c2[i] -= v2 * value;
            c3[i] -= v3 * value;
        }
    }

    private static void SolveLowerUnit1(int m, double* l, int ldl, double* column)
    {
        for (int p = 0; p < m - 1; p++)
        {
            double value = column[p];
            if (value != 0.0)
                Blas1.Axpy(m - p - 1, -value, l + (nint)p * ldl + p + 1, column + p + 1);
        }
    }

    private static void SolveUpper1(int m, double* u, int ldu, double* column)
    {
        for (int p = m - 1; p >= 0; p--)
        {
            column[p] /= u[(nint)p * ldu + p];

            double value = column[p];
            if (value != 0.0 && p > 0)
                Blas1.Axpy(p, -value, u + (nint)p * ldu, column);
        }
    }

    /// <summary>
    /// Multiply by a unit lower triangular factor: B := L * B, L m x m.
    /// Used only to rebuild P*A from the packed factors during verification.
    /// </summary>
    public static void MultiplyLowerUnit(int m, int n, double* l, int ldl, double* b, int ldb)
    {
        for (int j = 0; j < n; j++)
        {
            double* column = b + (nint)j * ldb;

            for (int i = m - 1; i >= 0; i--)
            {
                double sum = column[i];
                for (int p = 0; p < i; p++)
                    sum += l[(nint)p * ldl + i] * column[p];
                column[i] = sum;
            }
        }
    }
}
