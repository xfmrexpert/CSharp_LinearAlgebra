using System.Numerics;
using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// LU with partial pivoting, written once over the element type: P*A = L*U for
/// any <c>T</c> whose arithmetic an <see cref="IElementKernels{T}"/> supplies.
///
/// The algorithm is the same as the real kernel's (and LAPACK's getrf):
/// factor a tall panel unblocked, apply its interchanges either side, solve
/// L11 * U12 = A12, update A22 -= A21 * U12. The only level-3 step, the
/// trailing update, goes through <c>TKernels.Multiply</c> -- for complex that
/// is the 4M product on the workspace's GEMM, so the O(n^3) of the work runs
/// on the tuned real kernel. The rest (panel, swaps, triangular solve) is
/// O(n^2 * nb) and is ordinary safe code over spans.
///
/// The real path does not use this: it keeps the pointer-based factorization in
/// the kernel assembly, which is tuned and measured. Instantiated for
/// <see cref="double"/> this is instead that path's oracle: it must choose
/// exactly the same pivots, since it applies the same rule with the same
/// arithmetic. That comparison is what the tests lean on, and it covers the
/// complex instantiation too, because the only thing that differs between the
/// two is what <c>TKernels</c> supplies.
///
/// Pivots maximise <c>TKernels.PivotMagnitude</c>, which for complex is
/// |Re| + |Im| as in LAPACK's <c>izamax</c>, with ties to the lowest index as
/// in <c>idamax</c>. The pivot is applied as a reciprocal multiply unless its
/// modulus is below the smallest normal number, where the reciprocal would
/// overflow -- <c>zgetf2</c>'s rule, and the real kernel's.
/// </summary>
internal static class BlockedLu
{
    /// <summary>Smallest normalized double; below this, divide rather than multiply by a reciprocal.</summary>
    private const double SafeMin = 2.2250738585072014e-308;

    /// <summary>
    /// Factor <paramref name="a"/> in place. The result carries the pivots and
    /// diagnostics; the factors are left in <paramref name="a"/> in LAPACK's
    /// packed layout.
    /// </summary>
    /// <param name="workspace">Where the trailing updates run.</param>
    /// <param name="a">The m x n matrix, overwritten with the factors.</param>
    /// <param name="blockSize">Panel width; zero or less selects the default.</param>
    public static LuFactorization Factor<T, TKernels>(Workspace workspace, MatrixView<T> a, int blockSize)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int m = a.Rows, n = a.Columns;
        var result = new LuFactorization(m, n);
        int[] pivots = result.Pivots;

        int limit = Math.Min(m, n);
        if (limit == 0) return result;

        if (blockSize < 1) blockSize = Lu.DefaultBlockSizeFor(m, n);

        if (blockSize >= limit)
        {
            FactorPanel<T, TKernels>(a, pivots, 0, result);
            RecordPivotRange<T, TKernels>(a, result);
            return result;
        }

        for (int jb = 0; jb < limit; jb += blockSize)
        {
            int width = Math.Min(blockSize, limit - jb);

            FactorPanel<T, TKernels>(a.Slice(jb, jb, m - jb, width), pivots.AsSpan(jb, width), jb, result);

            // Panel pivots are relative to the panel top; make them global.
            for (int i = 0; i < width; i++) pivots[jb + i] += jb;

            // This panel's interchanges, applied to the columns either side of it.
            SwapRows(a, 0, jb, pivots, jb, jb + width);

            if (jb + width >= n) continue;

            SwapRows(a, jb + width, n, pivots, jb, jb + width);

            // U12 := L11^-1 * A12.
            SolveLowerUnit(
                a.Slice(jb, jb, width, width),
                a.Slice(jb, jb + width, width, n - jb - width));

            if (jb + width >= m) continue;

            // A22 := A22 - A21 * U12. The three views are disjoint blocks of
            // one matrix; the product reads A21 and U12 and writes only A22.
            TKernels.Multiply(
                workspace,
                a.Slice(jb + width, jb, m - jb - width, width),
                a.Slice(jb, jb + width, width, n - jb - width),
                a.Slice(jb + width, jb + width, m - jb - width, n - jb - width),
                -T.One,
                T.One);
        }

        RecordPivotRange<T, TKernels>(a, result);
        return result;
    }

    /// <summary>Solve A*X = B for a square factorization, overwriting <paramref name="b"/> with X.</summary>
    /// <exception cref="InvalidOperationException">The factorization has an exactly zero pivot.</exception>
    public static void Solve<T, TKernels>(LuFactorization lu, ReadOnlyMatrixView<T> factors, MatrixView<T> b)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        RequireSolvable(lu);

        int n = lu.Rows;
        if (n == 0 || b.Columns == 0) return;

        // P*B, then L*Y = P*B forward, then U*X = Y backward.
        SwapRows(b, 0, b.Columns, lu.Pivots, 0, n);
        SolveLowerUnit(factors, b);
        SolveUpper(factors, b);
    }

    /// <summary>
    /// Solve A^H*X = B for a square factorization, overwriting
    /// <paramref name="b"/> with X; A^T for a real type.
    ///
    /// P*A = L*U gives A^H = U^H * L^H * P, so this solves with U^H forward,
    /// then L^H backward, then undoes the permutation -- last and in reverse,
    /// where the plain solve applies it first and forward.
    /// </summary>
    /// <exception cref="InvalidOperationException">The factorization has an exactly zero pivot.</exception>
    public static void SolveAdjoint<T, TKernels>(LuFactorization lu, ReadOnlyMatrixView<T> factors, MatrixView<T> b)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        RequireSolvable(lu);

        int n = lu.Rows;
        if (n == 0 || b.Columns == 0) return;

        SolveUpperAdjoint<T, TKernels>(factors, b);
        SolveLowerUnitAdjoint<T, TKernels>(factors, b);
        UnswapRows(b, lu.Pivots, n);
    }

    /// <summary>
    /// Unblocked factorization of one panel (LAPACK's getf2). Pivots are
    /// written relative to the panel's own first row; a zero pivot is recorded
    /// against its column in the whole matrix.
    /// </summary>
    private static void FactorPanel<T, TKernels>(
        MatrixView<T> panel, Span<int> pivots, int columnOffset, LuFactorization result)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int m = panel.Rows, n = panel.Columns;
        int limit = Math.Min(m, n);

        for (int j = 0; j < limit; j++)
        {
            Span<T> column = panel.Column(j);

            int pivot = j + IndexOfLargestPivot<T, TKernels>(column[j..]);
            pivots[j] = pivot;

            T value = column[pivot];

            if (value == T.Zero)
            {
                if (result.SingularColumn < 0) result.SingularColumn = columnOffset + j;
            }
            else
            {
                if (pivot != j) SwapRowPair(panel, j, pivot);

                Span<T> below = column[(j + 1)..];

                if (TKernels.Magnitude(value) >= SafeMin)
                {
                    T reciprocal = T.One / value;
                    for (int i = 0; i < below.Length; i++) below[i] *= reciprocal;
                }
                else
                {
                    for (int i = 0; i < below.Length; i++) below[i] /= value;
                }
            }

            // Rank-1 update of the rest of the panel, a column at a time.
            ReadOnlySpan<T> multipliers = column[(j + 1)..];

            for (int jj = j + 1; jj < n; jj++)
            {
                Span<T> target = panel.Column(jj);
                T multiplier = target[j];
                if (multiplier == T.Zero) continue;

                T scale = -multiplier;
                Span<T> update = target[(j + 1)..];
                for (int i = 0; i < update.Length; i++) update[i] += scale * multipliers[i];
            }
        }
    }

    /// <summary>
    /// Index of the entry with the largest pivot measure, ties to the lowest
    /// index; 0 for an empty span. Strict comparison, so a NaN never wins and
    /// never displaces -- the same scan as the real kernel's.
    /// </summary>
    internal static int IndexOfLargestPivot<T, TKernels>(ReadOnlySpan<T> x)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        if (x.IsEmpty) return 0;

        int best = 0;
        double bestValue = TKernels.PivotMagnitude(x[0]);

        for (int i = 1; i < x.Length; i++)
        {
            double value = TKernels.PivotMagnitude(x[i]);
            if (value > bestValue)
            {
                bestValue = value;
                best = i;
            }
        }

        return best;
    }

    /// <summary>
    /// Apply interchanges pivots[first..last) to columns [columnStart,
    /// columnEnd), in increasing order (LAPACK's laswp). Column-outer, so each
    /// column is visited once with every interchange applied while it is in
    /// cache -- the order CLAUDE.md finding 4 measured.
    /// </summary>
    private static void SwapRows<T>(
        MatrixView<T> a, int columnStart, int columnEnd, ReadOnlySpan<int> pivots, int first, int last)
        where T : unmanaged, INumberBase<T>
    {
        for (int j = columnStart; j < columnEnd; j++)
        {
            Span<T> column = a.Column(j);

            for (int k = first; k < last; k++)
            {
                int pivot = pivots[k];
                if (pivot != k) (column[k], column[pivot]) = (column[pivot], column[k]);
            }
        }
    }

    /// <summary>Undo the first <paramref name="count"/> interchanges, walking them backwards.</summary>
    private static void UnswapRows<T>(MatrixView<T> a, ReadOnlySpan<int> pivots, int count)
        where T : unmanaged, INumberBase<T>
    {
        for (int j = 0; j < a.Columns; j++)
        {
            Span<T> column = a.Column(j);

            for (int k = count - 1; k >= 0; k--)
            {
                int pivot = pivots[k];
                if (pivot != k) (column[k], column[pivot]) = (column[pivot], column[k]);
            }
        }
    }

    /// <summary>Swap two rows across every column of <paramref name="a"/>.</summary>
    private static void SwapRowPair<T>(MatrixView<T> a, int rowA, int rowB)
        where T : unmanaged, INumberBase<T>
    {
        for (int j = 0; j < a.Columns; j++)
        {
            Span<T> column = a.Column(j);
            (column[rowA], column[rowB]) = (column[rowB], column[rowA]);
        }
    }

    /// <summary>
    /// B := L^-1 * B for the unit lower triangle of <paramref name="l"/>
    /// (its diagonal and upper part are not read). Column-oriented: each known
    /// x_k is subtracted down the column of L beneath it.
    /// </summary>
    private static void SolveLowerUnit<T>(ReadOnlyMatrixView<T> l, MatrixView<T> b)
        where T : unmanaged, INumberBase<T>
    {
        int n = b.Rows;

        for (int c = 0; c < b.Columns; c++)
        {
            Span<T> x = b.Column(c);

            for (int k = 0; k < n; k++)
            {
                T xk = x[k];
                if (xk == T.Zero) continue;

                T scale = -xk;
                ReadOnlySpan<T> below = l.Column(k)[(k + 1)..n];
                Span<T> target = x[(k + 1)..n];
                for (int i = 0; i < below.Length; i++) target[i] += scale * below[i];
            }
        }
    }

    /// <summary>B := U^-1 * B for the upper triangle of <paramref name="u"/>, diagonal included.</summary>
    private static void SolveUpper<T>(ReadOnlyMatrixView<T> u, MatrixView<T> b)
        where T : unmanaged, INumberBase<T>
    {
        int n = b.Rows;

        for (int c = 0; c < b.Columns; c++)
        {
            Span<T> x = b.Column(c);

            for (int k = n - 1; k >= 0; k--)
            {
                ReadOnlySpan<T> column = u.Column(k);

                if (x[k] == T.Zero) continue;

                x[k] /= column[k];

                T scale = -x[k];
                for (int i = 0; i < k; i++) x[i] += scale * column[i];
            }
        }
    }

    /// <summary>
    /// B := U^-H * B. U^H is lower triangular, so this runs forward, and each
    /// step is a dot product with a column of U above its diagonal -- the
    /// contiguous direction in column-major storage.
    /// </summary>
    private static void SolveUpperAdjoint<T, TKernels>(ReadOnlyMatrixView<T> u, MatrixView<T> b)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int n = b.Rows;

        for (int c = 0; c < b.Columns; c++)
        {
            Span<T> x = b.Column(c);

            for (int i = 0; i < n; i++)
            {
                ReadOnlySpan<T> column = u.Column(i);

                T sum = x[i];
                for (int k = 0; k < i; k++) sum -= TKernels.Conjugate(column[k]) * x[k];

                x[i] = sum / TKernels.Conjugate(column[i]);
            }
        }
    }

    /// <summary>
    /// B := L^-H * B for the unit lower triangle of L. L^H is unit upper, so
    /// this runs backward, a dot product with the column of L below each
    /// diagonal.
    /// </summary>
    private static void SolveLowerUnitAdjoint<T, TKernels>(ReadOnlyMatrixView<T> l, MatrixView<T> b)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int n = b.Rows;

        for (int c = 0; c < b.Columns; c++)
        {
            Span<T> x = b.Column(c);

            for (int i = n - 1; i >= 0; i--)
            {
                ReadOnlySpan<T> column = l.Column(i);

                T sum = x[i];
                for (int k = i + 1; k < n; k++) sum -= TKernels.Conjugate(column[k]) * x[k];

                x[i] = sum;
            }
        }
    }

    /// <summary>Smallest and largest modulus on U's diagonal, for the pivot ratio.</summary>
    private static void RecordPivotRange<T, TKernels>(ReadOnlyMatrixView<T> a, LuFactorization result)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int limit = Math.Min(a.Rows, a.Columns);
        if (limit == 0) return;

        double smallest = double.PositiveInfinity;
        double largest = 0.0;

        for (int j = 0; j < limit; j++)
        {
            double value = TKernels.Magnitude(a.Column(j)[j]);
            smallest = Math.Min(smallest, value);
            largest = Math.Max(largest, value);
        }

        result.SmallestPivot = smallest;
        result.LargestPivot = largest;
    }

    private static void RequireSolvable(LuFactorization lu)
    {
        if (!lu.IsSquare)
            throw new InvalidOperationException("Solve requires a square factorization.");

        if (lu.IsSingular)
            throw new InvalidOperationException($"Matrix is singular: zero pivot at column {lu.SingularColumn}.");
    }
}
