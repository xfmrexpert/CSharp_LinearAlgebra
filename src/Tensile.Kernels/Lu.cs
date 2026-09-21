using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Tensile.Kernels;

/// <summary>
/// Result of <see cref="Lu.Factor{TKernel}"/>: the pivot sequence and the
/// diagnostics. The factors themselves overwrite the caller's matrix (L below
/// the diagonal with an implicit unit diagonal, U on and above it) and are NOT
/// referenced from here -- every solve takes them as an explicit operand.
///
/// Holding no pointer is what makes this safe to keep: the caller's storage
/// may be pinned only for the duration of the factoring call, and a pointer
/// kept past that would be dangling the moment the pin was released. The
/// pivot array is an ordinary managed array, so there is nothing to dispose.
/// </summary>
internal sealed class LuFactorization
{
    /// <summary>Rows of the factored matrix.</summary>
    public int Rows { get; }

    /// <summary>Columns of the factored matrix.</summary>
    public int Columns { get; }

    /// <summary>
    /// Row interchanges, 0-based, length min(Rows, Columns). Entry k means row
    /// k was swapped with row Pivots[k] at step k. Applied in increasing k.
    /// </summary>
    public int[] Pivots { get; }

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

    /// <summary>Whether an exactly zero pivot was encountered. A solve would divide by zero.</summary>
    public bool IsSingular => SingularColumn >= 0;

    /// <summary>Whether the factored matrix was square, and so admits a solve.</summary>
    public bool IsSquare => Rows == Columns;

    internal LuFactorization(int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);

        Rows = rows;
        Columns = columns;
        Pivots = new int[Math.Min(rows, columns)];
    }
}

/// <summary>
/// Where a blocked factorization spends its time, by phase.
///
/// This exists because the split is load-bearing: the argument that LU's 45%
/// of threaded GEMM is Amdahl rather than a defect rests on how much of the
/// work is in the serial phases, and the figures it rested on were measured
/// single-core on a different machine. See CLAUDE.md, "LU".
///
/// Collection is opt-in and costs nothing when it is off. <see cref="Lu.Factor"/>
/// takes one of these or null; null means a handful of perfectly predictable
/// null checks per block step, which at n=2048 and nb=64 is 32 steps against
/// 88 ms of work. When it is on, the cost is one
/// <see cref="Stopwatch.GetTimestamp"/> per phase per block step — about 160
/// calls for that same factorization, a few microseconds, well under 0.01%.
/// tensile-diag reports that cost rather than asserting it: it factors each
/// size both ways and prints the difference.
///
/// <see cref="Total"/> is measured around the whole loop rather than summed
/// from the phases, so the difference between it and their sum is time the
/// phases do not account for — the pivot fix-up, the diagonal scan, loop
/// overhead. Reporting that residue is the point: a split that silently
/// normalised to 100% could hide it.
/// </summary>
internal sealed class LuPhaseTimings
{
    private long _panel;
    private long _swaps;
    private long _triangular;
    private long _gemm;

    /// <summary>Ticks in the unblocked panel factorization.</summary>
    public long Panel => _panel;

    /// <summary>Ticks applying row interchanges either side of the panel.</summary>
    public long Swaps => _swaps;

    /// <summary>Ticks in the triangular solve for U12.</summary>
    public long Triangular => _triangular;

    /// <summary>Ticks in the trailing-submatrix GEMM.</summary>
    public long Gemm => _gemm;

    /// <summary>Ticks for the whole blocked loop, phases and everything between.</summary>
    public long Total { get; private set; }

    /// <summary>Block steps taken.</summary>
    public int BlockSteps { get; private set; }

    /// <summary>Ticks the phases above do not account for.</summary>
    public long Unattributed => Math.Max(0, Total - (Panel + Swaps + Triangular + Gemm));

    /// <summary>A phase's share of the total, as a percentage.</summary>
    /// <param name="ticks">One of the phase totals.</param>
    public double Share(long ticks) => Total == 0 ? 0.0 : 100.0 * ticks / Total;

    /// <summary>Add another factorization's timings to these, for pooling over repetitions.</summary>
    /// <param name="other">The run to fold in.</param>
    public void Add(LuPhaseTimings other)
    {
        ArgumentNullException.ThrowIfNull(other);

        _panel += other._panel;
        _swaps += other._swaps;
        _triangular += other._triangular;
        _gemm += other._gemm;
        Total += other.Total;
        BlockSteps += other.BlockSteps;
    }

    /// <summary>A timestamp, or zero when <paramref name="timings"/> is null.</summary>
    /// <param name="timings">The collector, or null when collection is off.</param>
    public static long Now(LuPhaseTimings? timings) => timings is null ? 0L : Stopwatch.GetTimestamp();

    internal void AddPanel(ref long mark) => Accumulate(ref _panel, ref mark);
    internal void AddSwaps(ref long mark) => Accumulate(ref _swaps, ref mark);
    internal void AddTriangular(ref long mark) => Accumulate(ref _triangular, ref mark);
    internal void AddGemm(ref long mark) => Accumulate(ref _gemm, ref mark);

    internal void AddStep() => BlockSteps++;

    internal void AddTotal(long start) => Total += Stopwatch.GetTimestamp() - start;

    /// <summary>
    /// Charge the time since <paramref name="mark"/> to a bucket and re-mark.
    /// The bucket is taken by reference rather than through a delegate so that
    /// the enabled path allocates nothing at all: a closure here would be a
    /// delegate per phase per block step, which is exactly the sort of cost a
    /// measurement instrument must not add to what it measures.
    /// </summary>
    private static void Accumulate(ref long bucket, ref long mark)
    {
        long now = Stopwatch.GetTimestamp();
        bucket += now - mark;
        mark = now;
    }
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
internal static unsafe class Lu
{
    /// <summary>Smallest normalized double; below this, divide rather than multiply by a reciprocal.</summary>
    private const double SafeMin = 2.2250738585072014e-308;

    /// <summary>
    /// Default panel width for a matrix too large for <see cref="SmallBlockSize"/>.
    /// </summary>
    public const int DefaultBlockSize = 64;

    /// <summary>Panel width below <see cref="SizeBlockSizeCrossover"/>.</summary>
    public const int SmallBlockSize = 32;

    /// <summary>
    /// Order at or above which <see cref="DefaultBlockSize"/> takes over from
    /// <see cref="SmallBlockSize"/>. Measured between, not derived: nb=32 wins
    /// at n=512 and nb=64 wins at n=1024, so the true crossover is somewhere
    /// in between and 1024 is the conservative end of that interval.
    /// </summary>
    public const int SizeBlockSizeCrossover = 1024;

    /// <summary>
    /// The panel width to use for an m x n matrix when the caller does not
    /// choose one.
    ///
    /// Measured on a 12700H, both sweep directions agreeing at every size that
    /// matters here. nb=32 beats nb=64 by 11.0% at n=256 and 12.6% at n=512;
    /// nb=64 beats nb=32 by 4.6% at n=1024; at n=2048 the two are within 0.6%
    /// and the directions disagree on which leads, so either is fine there.
    /// nb=128 was worst at every size in both directions, by 19-33%, which has
    /// a mechanism behind it as well as a measurement: the panel is O(m*nb^2)
    /// level-2 work, so a wider panel moves more of the total into the slow
    /// unblocked path.
    ///
    /// The small-n results are the ones worth trusting most, oddly enough:
    /// they hold up in the descending sweep, where nb=32 runs last and
    /// hottest and wins anyway. See CLAUDE.md finding 7.
    /// </summary>
    /// <param name="rows">Rows of the matrix to factor.</param>
    /// <param name="columns">Columns of the matrix to factor.</param>
    public static int DefaultBlockSizeFor(int rows, int columns) =>
        Math.Min(rows, columns) < SizeBlockSizeCrossover ? SmallBlockSize : DefaultBlockSize;

    /// <summary>
    /// Factor the m x n column-major matrix in place as P*A = L*U. The result
    /// holds the pivots and diagnostics only; the factors are left in
    /// <paramref name="a"/>, and the solves take them back as an argument.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static LuFactorization Factor<TKernel>(
        int m, int n, double* a, int lda, GemmDispatch gemm, int blockSize = 0,
        LuPhaseTimings? timings = null)
        where TKernel : struct, IMicroKernel
    {
        var result = new LuFactorization(m, n);
        int[] pivots = result.Pivots;

        int limit = Math.Min(m, n);
        if (limit == 0) return result;

        if (blockSize < 1) blockSize = DefaultBlockSizeFor(m, n);

        // Small problems are all panel and no GEMM; skip the blocking machinery.
        if (blockSize >= limit)
        {
            FactorPanel(m, n, a, lda, pivots, 0, result);
            RecordPivotRange(result, a, lda);
            return result;
        }

        long loopStart = LuPhaseTimings.Now(timings);

        for (int jb = 0; jb < limit; jb += blockSize)
        {
            int width = Math.Min(blockSize, limit - jb);
            double* panel = a + (nint)jb * lda + jb;
            long mark = LuPhaseTimings.Now(timings);

            timings?.AddStep();

            // Factor the current panel: rows jb..m-1, columns jb..jb+width-1.
            FactorPanel(m - jb, width, panel, lda, pivots.AsSpan(jb, width), jb, result);

            // Panel pivots are relative to the panel top; make them global.
            for (int i = 0; i < width; i++) pivots[jb + i] += jb;

            if (timings is not null) timings.AddPanel(ref mark);

            // Apply this panel's interchanges to the columns either side of it.
            SwapRows(a, lda, 0, jb, pivots, jb, jb + width);

            if (jb + width >= n)
            {
                if (timings is not null) timings.AddSwaps(ref mark);
                continue;
            }

            SwapRows(a, lda, jb + width, n, pivots, jb, jb + width);

            if (timings is not null) timings.AddSwaps(ref mark);

            // U12 := L11^-1 * A12.
            Triangular.SolveLowerUnit(
                width, n - jb - width,
                a + (nint)jb * lda + jb, lda,
                a + (nint)(jb + width) * lda + jb, lda);

            if (timings is not null) timings.AddTriangular(ref mark);

            if (jb + width >= m) continue;

            // A22 := A22 - A21 * U12. This is the blocked part, and the only
            // level-3 operation in the factorization, so it is the one worth
            // threading. The dispatch drops back to the serial path by itself
            // once the trailing submatrix stops being worth a fork/join.
            gemm.Multiply<TKernel>(
                m - jb - width, n - jb - width, width,
                -1.0, a + (nint)jb * lda + (jb + width), lda,
                a + (nint)(jb + width) * lda + jb, lda,
                1.0, a + (nint)(jb + width) * lda + (jb + width), lda);

            if (timings is not null) timings.AddGemm(ref mark);
        }

        timings?.AddTotal(loopStart);

        RecordPivotRange(result, a, lda);
        return result;
    }

    /// <summary>Scan U's diagonal so callers can spot a near-singular factorization.</summary>
    private static void RecordPivotRange(LuFactorization result, double* a, int lda)
    {
        int limit = Math.Min(result.Rows, result.Columns);
        if (limit == 0) return;

        double smallest = double.PositiveInfinity;
        double largest = 0.0;

        for (int j = 0; j < limit; j++)
        {
            double value = Math.Abs(a[(nint)j * lda + j]);
            smallest = Math.Min(smallest, value);
            largest = Math.Max(largest, value);
        }

        result.SmallestPivot = smallest;
        result.LargestPivot = largest;
    }

    /// <summary>
    /// Solve A*X = B for an already-factored square A. <paramref name="factors"/>
    /// is the matrix <see cref="Factor{TKernel}"/> overwrote, with the stride
    /// it had then; B is n x nrhs, column-major, overwritten with the solution.
    /// </summary>
    public static void Solve(LuFactorization lu, double* factors, int lda, int nrhs, double* b, int ldb)
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
        Triangular.SolveLowerUnit(n, nrhs, factors, lda, b, ldb);
        Triangular.SolveUpper(n, nrhs, factors, lda, b, ldb);
    }

    /// <summary>
    /// Solve A^T * X = B for an already-factored square A. B is n x nrhs,
    /// column-major, overwritten with the solution.
    ///
    /// P*A = L*U gives A = P^T*L*U and so A^T = U^T*L^T*P. The permutation
    /// therefore lands last and in reverse, rather than first and forward as it
    /// does in the untransposed solve.
    ///
    /// Needed by the 1-norm condition estimator, which alternates products with
    /// A^-1 and A^-T.
    /// </summary>
    public static void SolveTransposed(LuFactorization lu, double* factors, int lda, int nrhs, double* b, int ldb)
    {
        if (!lu.IsSquare)
            throw new ArgumentException("SolveTransposed requires a square factorization.", nameof(lu));
        if (lu.IsSingular)
            throw new InvalidOperationException(
                $"Matrix is singular: zero pivot at column {lu.SingularColumn}.");

        int n = lu.Rows;
        if (n == 0 || nrhs == 0) return;

        Triangular.SolveUpperTransposed(n, nrhs, factors, lda, b, ldb);
        Triangular.SolveLowerUnitTransposed(n, nrhs, factors, lda, b, ldb);
        UnswapRows(b, ldb, 0, nrhs, lu.Pivots, 0, n);
    }

    /// <summary>
    /// Unblocked level-2 factorization of one panel (LAPACK's dgetf2).
    /// Pivots are written relative to the panel's own first row.
    /// </summary>
    private static void FactorPanel(
        int m, int n, double* a, int lda, Span<int> pivots, int columnOffset, LuFactorization result)
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
        double* a, int lda, int columnStart, int columnEnd, ReadOnlySpan<int> pivots, int first, int last)
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
        double* a, int lda, int columnStart, int columnEnd, ReadOnlySpan<int> pivots, int first, int last)
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
