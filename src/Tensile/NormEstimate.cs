namespace Tensile;

/// <summary>Outcome of a 1-norm estimate.</summary>
/// <param name="Value">The estimate, always a lower bound on the true 1-norm.</param>
/// <param name="Iterations">Iterations performed.</param>
/// <param name="Products">Operator applications, counting each probe column.</param>
public readonly record struct NormEstimateResult(double Value, int Iterations, int Products);

/// <summary>
/// Block 1-norm estimation, after Higham and Tisseur, "A block algorithm for
/// matrix 1-norm estimation, with an application to 1-norm pseudospectra",
/// SIAM J. Matrix Anal. Appl. 21(4):1185-1201, 2000. This is the algorithm
/// behind MATLAB's normest1 and the block generalisation of LAPACK's dlacn2.
///
/// The estimate is obtained from a few products with A and A^T rather than from
/// A's entries, which is what makes it useful twice over: with
/// <see cref="LuInverseOperator"/> it gives a dgecon-equivalent condition
/// estimate without forming A^-1, and with <see cref="DenseMatrixOperator"/>
/// raised to a power it gives the ||A^k||^(1/k) quantities that Al-Mohy and
/// Higham's scaling-and-squaring uses to choose its scaling parameter.
///
/// Two properties matter for callers:
///
/// - The result is always a LOWER bound on the true 1-norm. It is exact in the
///   large majority of cases and rarely off by more than a factor of 2, but it
///   can underestimate, and no run-time signal distinguishes the two. A
///   condition estimate built on it is therefore optimistic, exactly as
///   dgecon's is.
/// - It is deterministic for a given seed. The algorithm needs random +/-1
///   starting vectors, but a fixed default seed means repeated runs on the same
///   matrix agree, which is what makes it usable in a regression test.
///
/// The algorithm is bookkeeping around the operator's products -- sign
/// matrices, column norms, a sort -- and every buffer it needs is O(n*t), so it
/// lives in the public assembly as ordinary safe code over managed arrays. The
/// O(n^2*t) work is the operator's, and a dense operator does it in the kernel
/// assembly.
/// </summary>
public static class NormEstimate
{
    /// <summary>Probe columns. Higham and Tisseur recommend 2; more costs more products.</summary>
    public const int DefaultColumns = 2;

    /// <summary>Iteration cap. MATLAB's normest1 uses 5; convergence is usually in 2 or 3.</summary>
    public const int DefaultMaxIterations = 5;

    /// <summary>Fixed so that estimates are reproducible run to run.</summary>
    public const int DefaultSeed = 20260915;

    /// <summary>Attempts before accepting a probe column parallel to an earlier one.</summary>
    private const int ResampleLimit = 8;

    /// <summary>Estimate ||A||_1 for an arbitrary square operator.</summary>
    /// <param name="op">The operator to probe.</param>
    /// <param name="columns">Probe columns; clamped to [1, n]. More costs more products and estimates better.</param>
    /// <param name="maxIterations">Iteration cap; at least 2 is used.</param>
    /// <param name="seed">Seed for the random starting probes.</param>
    /// <exception cref="ArgumentOutOfRangeException">The operator's order is negative, or the n x t probe panel would not fit a buffer.</exception>
    /// <exception cref="AllocationLimitException">A probe panel would exceed <see cref="TensileLimits.MaxElements"/>.</exception>
    public static NormEstimateResult Of(
        ILinearOperator op,
        int columns = DefaultColumns,
        int maxIterations = DefaultMaxIterations,
        int seed = DefaultSeed)
    {
        ArgumentNullException.ThrowIfNull(op);

        int n = op.Order;

        if (n < 0)
            throw new ArgumentOutOfRangeException(nameof(op), $"The operator reports a negative order, {n}.");

        if (n == 0) return new NormEstimateResult(0.0, 0, 0);

        int t = Math.Clamp(columns, 1, n);

        // Higham and Tisseur's convergence tests compare against the previous
        // iteration, so fewer than two iterations cannot terminate meaningfully.
        int itmax = Math.Max(maxIterations, 2);

        // n * t is the probe panel. An operator is free to report any Order,
        // and a matrix-free one has no storage whose size would constrain it,
        // so the extent is computed in long and checked before anything is
        // allocated: with n = t = 65536 the int product wraps to exactly zero.
        // This assembly compiles with overflow checking on, so even an unguarded
        // product would throw rather than wrap -- but an OverflowException is
        // the wrong diagnosis for what is an argument error, hence the check.
        if ((long)n * t > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(columns),
                $"An order-{n} operator with {t} probe columns needs a {(long)n * t}-element panel, "
                + $"which exceeds the {int.MaxValue} a buffer can address.");
        }

        var panel = new MatrixShape(n, t);
        int extent = panel.RequiredExtent;

        // Five n x t panels and a handful of length-n vectors. The policy
        // limit applies per allocation, so it is the panel that is measured
        // against it; an operator whose order alone exceeded the limit could
        // not have been built from a matrix, but a matrix-free one can report
        // any order it likes.
        string panelPurpose = $"an {n}x{t} probe panel";
        string vectorPurpose = $"an order-{n} work vector";

        double[] x = Storage.Array<double>(extent, panelPurpose);
        double[] y = Storage.Array<double>(extent, panelPurpose);
        double[] s = Storage.Array<double>(extent, panelPurpose);
        double[] sOld = Storage.Array<double>(extent, panelPurpose);
        double[] z = Storage.Array<double>(extent, panelPurpose);
        double[] h = Storage.Array<double>(n, vectorPurpose);

        // Not a security use of randomness (CA5394 is off for this assembly,
        // see .editorconfig): the signs only have to be uncorrelated with the
        // matrix, and a fixed seed is what makes the estimate reproducible.
        var rng = new Random(seed);
        int[] order = Storage.Array<int>(n, vectorPurpose);
        int[] scratchOrder = Storage.Array<int>(n, vectorPurpose);
        double[] keys = Storage.Array<double>(n, vectorPurpose);
        bool[] used = Storage.Array<bool>(n, vectorPurpose);
        int[] current = Storage.Array<int>(t, vectorPurpose);

        InitialProbe(rng, n, t, x);

        double est = 0.0;
        double estOld = 0.0;
        int bestIndex = -1;
        int iterations = 0;
        int products = 0;

        for (int k = 1; ; k++)
        {
            iterations = k;

            op.Apply(ReadOnlyMatrixView<double>.Bind(x, panel), MatrixView<double>.Bind(y, panel));
            products += t;

            int argmax = 0;
            est = ColumnOneNorm(y, 0, n);

            for (int j = 1; j < t; j++)
            {
                double value = ColumnOneNorm(y, j * n, n);
                if (value > est) { est = value; argmax = j; }
            }

            // The unit vector behind the best column becomes the reference
            // point for the h test below. Only meaningful from k=2, when the
            // columns of X are unit vectors chosen by the previous iteration.
            if (k >= 2 && (est > estOld || k == 2)) bestIndex = current[argmax];

            // Converged: this iteration did not improve on the last.
            if (k >= 2 && est <= estOld)
            {
                est = estOld;
                break;
            }

            estOld = est;

            // Higham and Tisseur stop at k > itmax, not k = itmax: the last
            // iteration still gets to compute Y and improve the estimate, it
            // just does not get to choose a new probe.
            if (k > itmax) break;

            // S := sign(Y), taking sign(0) as +1. The previous S is kept to
            // detect a repeat, which is the other termination test.
            (s, sOld) = (sOld, s);
            for (int i = 0; i < extent; i++) s[i] = y[i] >= 0.0 ? 1.0 : -1.0;

            if (t > 1)
            {
                if (AllColumnsParallel(n, t, s, sOld)) break;
                MakeColumnsDistinct(rng, n, t, s, sOld);
            }

            op.ApplyTranspose(ReadOnlyMatrixView<double>.Bind(s, panel), MatrixView<double>.Bind(z, panel));
            products += t;

            for (int i = 0; i < n; i++)
            {
                double best = 0.0;
                for (int j = 0; j < t; j++)
                    best = Math.Max(best, Math.Abs(z[j * n + i]));
                h[i] = best;
            }

            // No index beats the one already chosen, so no further probe can
            // improve the estimate.
            if (k >= 2 && bestIndex >= 0)
            {
                double largest = 0.0;
                for (int i = 0; i < n; i++) largest = Math.Max(largest, h[i]);
                if (largest == h[bestIndex]) break;
            }

            SortIndicesDescending(n, h, keys, order);

            if (t > 1)
            {
                bool allUsed = true;
                for (int j = 0; j < t; j++)
                {
                    if (!used[order[j]]) { allUsed = false; break; }
                }

                // Every candidate has already been probed; repeating them
                // would cycle.
                if (allUsed) break;

                MoveUnusedToFront(n, order, used, scratchOrder);
            }

            Array.Clear(x);

            for (int j = 0; j < t; j++)
            {
                int index = order[j];
                current[j] = index;
                used[index] = true;
                x[j * n + index] = 1.0;
            }
        }

        return new NormEstimateResult(est, iterations, products);
    }

    /// <summary>
    /// Starting probe: one column of ones, the rest random +/-1 and distinct
    /// from their predecessors, all scaled to unit 1-norm.
    /// </summary>
    private static void InitialProbe(Random rng, int n, int t, double[] x)
    {
        for (int i = 0; i < n; i++) x[i] = 1.0;

        for (int j = 1; j < t; j++)
        {
            for (int attempt = 0; attempt < ResampleLimit; attempt++)
            {
                FillRandomSigns(rng, x, j * n, n);
                if (!IsParallelToAny(n, x, j * n, x, j)) break;
            }
        }

        for (int i = 0; i < n * t; i++) x[i] /= n;
    }

    private static void FillRandomSigns(Random rng, double[] panel, int start, int n)
    {
        for (int i = 0; i < n; i++) panel[start + i] = rng.Next(2) == 0 ? -1.0 : 1.0;
    }

    /// <summary>
    /// Two +/-1 vectors are parallel exactly when |u.v| = n. The entries are
    /// integers well inside the exactly representable range, so this comparison
    /// is safe.
    /// </summary>
    private static bool IsParallel(int n, ReadOnlySpan<double> u, ReadOnlySpan<double> v)
    {
        double dot = 0.0;
        for (int i = 0; i < n; i++) dot += u[i] * v[i];
        return Math.Abs(dot) == n;
    }

    /// <summary>Whether the column at <paramref name="start"/> is parallel to any of the first <paramref name="count"/> columns of <paramref name="panel"/>.</summary>
    private static bool IsParallelToAny(int n, double[] column, int start, double[] panel, int count)
    {
        ReadOnlySpan<double> candidate = column.AsSpan(start, n);

        for (int j = 0; j < count; j++)
        {
            if (IsParallel(n, candidate, panel.AsSpan(j * n, n))) return true;
        }

        return false;
    }

    /// <summary>Every column of S has already appeared in S_old.</summary>
    private static bool AllColumnsParallel(int n, int t, double[] s, double[] sOld)
    {
        for (int j = 0; j < t; j++)
        {
            if (!IsParallelToAny(n, s, j * n, sOld, t)) return false;
        }

        return true;
    }

    /// <summary>
    /// Resample any column of S that duplicates an earlier column of S or a
    /// column of S_old. Duplicates waste a product without adding information.
    /// </summary>
    private static void MakeColumnsDistinct(Random rng, int n, int t, double[] s, double[] sOld)
    {
        for (int j = 0; j < t; j++)
        {
            for (int attempt = 0; attempt < ResampleLimit; attempt++)
            {
                if (!IsParallelToAny(n, s, j * n, s, j) && !IsParallelToAny(n, s, j * n, sOld, t))
                    break;

                FillRandomSigns(rng, s, j * n, n);
            }
        }
    }

    private static void SortIndicesDescending(int n, double[] h, double[] keys, int[] order)
    {
        for (int i = 0; i < n; i++)
        {
            keys[i] = -h[i];
            order[i] = i;
        }

        Array.Sort(keys, order);
    }

    /// <summary>
    /// Stable partition of <paramref name="order"/> putting not-yet-probed
    /// indices first, so the next round spends its columns on new information.
    /// </summary>
    private static void MoveUnusedToFront(int n, int[] order, bool[] used, int[] scratch)
    {
        int at = 0;

        for (int i = 0; i < n; i++)
        {
            if (!used[order[i]]) scratch[at++] = order[i];
        }

        for (int i = 0; i < n; i++)
        {
            if (used[order[i]]) scratch[at++] = order[i];
        }

        Array.Copy(scratch, order, n);
    }

    private static double ColumnOneNorm(double[] panel, int start, int n)
    {
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += Math.Abs(panel[start + i]);
        return sum;
    }
}

/// <summary>
/// Condition estimation in the 1-norm, the equivalent of LAPACK's dgecon.
/// Reached through <see cref="LuDecomposition.ReciprocalCondition"/>, which
/// supplies the norm of the original matrix itself.
/// </summary>
internal static class Condition
{
    /// <summary>
    /// Estimate 1/cond_1(A) = 1/(||A||_1 * ||A^-1||_1) from a factorization.
    ///
    /// <paramref name="normOfA"/> is ||A||_1 of the ORIGINAL matrix, captured
    /// before factoring overwrote it. Returns 0 for an exactly singular
    /// factorization. The estimator underestimates ||A^-1||_1, so the value
    /// returned is an OVER-estimate of the reciprocal condition number: a small
    /// result reliably means ill-conditioning, a large one is weaker evidence
    /// of good conditioning. This is the same asymmetry dgecon has, and the
    /// reason <see cref="LuDecomposition.PivotRatio"/> is not a substitute.
    /// </summary>
    public static double ReciprocalOne(
        double normOfA,
        LuDecomposition lu,
        int columns = NormEstimate.DefaultColumns,
        int maxIterations = NormEstimate.DefaultMaxIterations,
        int seed = NormEstimate.DefaultSeed)
    {
        ArgumentNullException.ThrowIfNull(lu);

        if (lu.IsSingular || normOfA == 0.0 || lu.Rows == 0) return 0.0;

        var op = new LuInverseOperator(lu);
        double inverseNorm = NormEstimate.Of(op, columns, maxIterations, seed).Value;

        return inverseNorm == 0.0 ? 0.0 : 1.0 / (normOfA * inverseNorm);
    }
}
