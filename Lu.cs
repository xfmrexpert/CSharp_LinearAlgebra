using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace GemmLab;

/// <summary>
/// Result of <see cref="Lu.Factor{TKernel}"/>. The factors overwrite the
/// caller's matrix (L below the diagonal with an implicit unit diagonal, U on
/// and above it); only the pivot array is owned here.
/// </summary>
public sealed unsafe class LuFactorization : IDisposable
{
    public int Rows { get; }
    public int Columns { get; }
    public int Stride { get; }

    /// <summary>The caller's buffer, factored in place. Not owned.</summary>
    public double* Factors { get; }

    /// <summary>
    /// Row interchanges, 0-based, length min(Rows, Columns). Entry k means row
    /// k was swapped with row Pivots[k] at step k. Applied in increasing k.
    /// </summary>
    public int* Pivots { get; private set; }

    /// <summary>
    /// Column index of the first exactly-zero pivot, or -1 if none. Mirrors
    /// LAPACK's info: factorization still completes, but a solve would divide
    /// by zero.
    /// </summary>
    public int SingularColumn { get; internal set; } = -1;

    /// <summary>Smallest magnitude on U's diagonal. Zero implies exact singularity.</summary>
    public double SmallestPivot { get; internal set; }

    /// <summary>Largest magnitude on U's diagonal.</summary>
    public double LargestPivot { get; internal set; }

    /// <summary>
    /// SmallestPivot / LargestPivot. A crude conditioning indicator, NOT a
    /// condition number: it can be optimistic by orders of magnitude. Use it to
    /// notice trouble, not to certify its absence -- that needs a proper
    /// 1-norm condition estimator.
    /// </summary>
    public double PivotRatio => LargestPivot == 0.0 ? 0.0 : SmallestPivot / LargestPivot;

    public bool IsSingular => SingularColumn >= 0;
    public bool IsSquare => Rows == Columns;

    internal LuFactorization(int rows, int columns, int stride, double* factors)
    {
        Rows = rows;
        Columns = columns;
        Stride = stride;
        Factors = factors;

        int count = Math.Max(1, Math.Min(rows, columns));
        Pivots = (int*)NativeMemory.AlignedAlloc((nuint)count * sizeof(int), 64);
        new Span<int>(Pivots, count).Clear();
    }

    public void Dispose()
    {
        if (Pivots is not null) { NativeMemory.AlignedFree(Pivots); Pivots = null; }
        GC.SuppressFinalize(this);
    }

    ~LuFactorization() => Dispose();
}

/// <summary>
/// LU factorization with partial pivoting, right-looking and blocked so that
/// the trailing-submatrix update is a GEMM.
///
/// The algorithm is LAPACK's dgetrf in structure: factor a tall panel with the
/// unblocked level-2 routine, apply the resulting row interchanges to the rest
/// of the matrix, solve L11 * U12 = A12, then update A22 -= A21 * U12. The
/// GEMM carries roughly 1 - 3*nb/(2*n) of the flops, so at n=1024 and nb=64
/// about 91% of the work runs at the speed measured in the GEMM benchmarks.
///
/// Partial pivoting is not backward stable in the strict sense — the growth
/// factor can reach 2^(n-1) on adversarial matrices — but it is stable in
/// practice, and it is what every production library uses. Check the reported
/// residual rather than assuming.
/// </summary>
public static unsafe class Lu
{
    /// <summary>Smallest normalized double; below this, divide rather than multiply by a reciprocal.</summary>
    private const double SafeMin = 2.2250738585072014e-308;

    public const int DefaultBlockSize = 64;

    /// <summary>
    /// Factor the m x n column-major matrix in place as P*A = L*U.
    /// The returned object must be disposed; it does not own the matrix.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static LuFactorization Factor<TKernel>(
        int m, int n, double* a, int lda, GemmScratch scratch, int blockSize = DefaultBlockSize)
        where TKernel : struct, IMicroKernel
    {
        var result = new LuFactorization(m, n, lda, a);

        int limit = Math.Min(m, n);
        if (limit == 0) return result;

        if (blockSize < 1) blockSize = DefaultBlockSize;

        // Small problems are all panel and no GEMM; skip the blocking machinery.
        if (blockSize >= limit)
        {
            FactorPanel(m, n, a, lda, result.Pivots, 0, result);
            RecordPivotRange(result);
            return result;
        }

        for (int jb = 0; jb < limit; jb += blockSize)
        {
            int width = Math.Min(blockSize, limit - jb);
            double* panel = a + (nint)jb * lda + jb;

            // Factor the current panel: rows jb..m-1, columns jb..jb+width-1.
            FactorPanel(m - jb, width, panel, lda, result.Pivots + jb, jb, result);

            // Panel pivots are relative to the panel top; make them global.
            for (int i = 0; i < width; i++) result.Pivots[jb + i] += jb;

            // Apply this panel's interchanges to the columns either side of it.
            SwapRows(a, lda, 0, jb, result.Pivots, jb, jb + width);

            if (jb + width >= n) continue;

            SwapRows(a, lda, jb + width, n, result.Pivots, jb, jb + width);

            // U12 := L11^-1 * A12.
            Triangular.SolveLowerUnit(
                width, n - jb - width,
                a + (nint)jb * lda + jb, lda,
                a + (nint)(jb + width) * lda + jb, lda);

            if (jb + width >= m) continue;

            // A22 := A22 - A21 * U12. This is the blocked part.
            Gemm.Multiply<TKernel>(
                m - jb - width, n - jb - width, width,
                -1.0, a + (nint)jb * lda + (jb + width), lda,
                a + (nint)(jb + width) * lda + jb, lda,
                1.0, a + (nint)(jb + width) * lda + (jb + width), lda,
                scratch);
        }

        RecordPivotRange(result);
        return result;
    }

    /// <summary>Scan U's diagonal so callers can spot a near-singular factorization.</summary>
    private static void RecordPivotRange(LuFactorization result)
    {
        int limit = Math.Min(result.Rows, result.Columns);
        if (limit == 0) return;

        double smallest = double.PositiveInfinity;
        double largest = 0.0;

        for (int j = 0; j < limit; j++)
        {
            double value = Math.Abs(result.Factors[(nint)j * result.Stride + j]);
            smallest = Math.Min(smallest, value);
            largest = Math.Max(largest, value);
        }

        result.SmallestPivot = smallest;
        result.LargestPivot = largest;
    }

    /// <summary>
    /// Solve A*X = B for an already-factored square A. B is n x nrhs,
    /// column-major, overwritten with the solution.
    /// </summary>
    public static void Solve(LuFactorization lu, int nrhs, double* b, int ldb)
    {
        if (!lu.IsSquare)
            throw new ArgumentException("Solve requires a square factorization.", nameof(lu));
        if (lu.IsSingular)
            throw new InvalidOperationException(
                $"Matrix is singular: zero pivot at column {lu.SingularColumn}.");

        int n = lu.Rows;
        if (n == 0 || nrhs == 0) return;

        // P*b, then forward substitution, then back substitution.
        SwapRows(b, ldb, 0, nrhs, lu.Pivots, 0, n);
        Triangular.SolveLowerUnit(n, nrhs, lu.Factors, lu.Stride, b, ldb);
        Triangular.SolveUpper(n, nrhs, lu.Factors, lu.Stride, b, ldb);
    }

    /// <summary>
    /// Unblocked level-2 factorization of one panel (LAPACK's dgetf2).
    /// Pivots are written relative to the panel's own first row.
    /// </summary>
    private static void FactorPanel(
        int m, int n, double* a, int lda, int* pivots, int columnOffset, LuFactorization result)
    {
        int limit = Math.Min(m, n);

        for (int j = 0; j < limit; j++)
        {
            double* column = a + (nint)j * lda;

            int pivot = j + Blas1.IndexOfMaxAbs(m - j, column + j);
            pivots[j] = pivot;

            double value = column[pivot];

            if (value == 0.0)
            {
                if (result.SingularColumn < 0)
                    result.SingularColumn = columnOffset + j;
            }
            else
            {
                if (pivot != j) SwapRowPair(a, lda, n, j, pivot);

                // Reciprocal multiply is faster, but underflows for a
                // denormal pivot -- fall back to division there.
                if (Math.Abs(value) >= SafeMin)
                    Blas1.Scale(m - j - 1, 1.0 / value, column + j + 1);
                else
                    for (int i = j + 1; i < m; i++) column[i] /= value;
            }

            // Rank-1 update of the trailing panel.
            for (int jj = j + 1; jj < n; jj++)
            {
                double* target = a + (nint)jj * lda;
                double multiplier = target[j];
                if (multiplier != 0.0)
                    Blas1.Axpy(m - j - 1, -multiplier, column + j + 1, target + j + 1);
            }
        }
    }

    /// <summary>
    /// Apply interchanges pivots[first..last) to columns [columnStart, columnEnd)
    /// of a column-major matrix, in increasing pivot order (LAPACK's dlaswp).
    /// </summary>
    internal static void SwapRows(
        double* a, int lda, int columnStart, int columnEnd, int* pivots, int first, int last)
    {
        // Column-outer, pivot-inner. The transposed order re-streams the whole
        // column range once per interchange; measured at 21% of LU runtime at
        // n=2048. This touches each column once and applies every interchange
        // while it is still in cache, for 9%. Swaps on different columns are
        // independent, so the permutation is unchanged.
        for (int j = columnStart; j < columnEnd; j++)
        {
            double* column = a + (nint)j * lda;

            for (int k = first; k < last; k++)
            {
                int pivot = pivots[k];
                if (pivot != k) (column[k], column[pivot]) = (column[pivot], column[k]);
            }
        }
    }

    /// <summary>Swap two rows across the first <paramref name="n"/> columns.</summary>
    private static void SwapRowPair(double* a, int lda, int n, int rowA, int rowB)
    {
        for (int j = 0; j < n; j++)
        {
            double* column = a + (nint)j * lda;
            (column[rowA], column[rowB]) = (column[rowB], column[rowA]);
        }
    }

    /// <summary>
    /// Undo the row permutation: given X holding L*U, produce P^-1 * X = A.
    /// Interchanges are reversed, so this walks the pivot array backwards.
    /// Verification only.
    /// </summary>
    internal static void UnswapRows(
        double* a, int lda, int columnStart, int columnEnd, int* pivots, int first, int last)
    {
        for (int j = columnStart; j < columnEnd; j++)
        {
            double* column = a + (nint)j * lda;

            for (int k = last - 1; k >= first; k--)
            {
                int pivot = pivots[k];
                if (pivot != k) (column[k], column[pivot]) = (column[pivot], column[k]);
            }
        }
    }
}
