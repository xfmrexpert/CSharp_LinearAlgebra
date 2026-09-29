using System.Numerics;

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
/// <see cref="LuInverseOperator{T}"/> it gives a dgecon-equivalent condition
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
/// The complex estimator is the same algorithm with three substitutions, as in
/// Higham and Tisseur's complex variant and MATLAB's normest1: the adjoint
/// A^H where the real one uses A^T; sign(y) = y/|y| (and 1 at zero), a
/// unit-modulus direction rather than +/-1; and no test for parallel sign
/// columns, which is meaningful only when the signs are a discrete set -- a
/// complex sign vector almost never repeats exactly, and resampling one would
/// buy nothing. The starting probes are the real +/-1 ones in both cases.
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
        IAdjointOperator<double> op,
        int columns = DefaultColumns,
        int maxIterations = DefaultMaxIterations,
        int seed = DefaultSeed) =>
        Estimate<double, DoubleKernels>(op, columns, maxIterations, seed);

    /// <summary>
    /// Estimate ||A||_1 for an arbitrary square complex operator, where
    /// ||A||_1 is the largest column sum of moduli. The operator's adjoint is
    /// the conjugate transpose.
    /// </summary>
    /// <param name="op">The operator to probe.</param>
    /// <param name="columns">Probe columns; clamped to [1, n]. More costs more products and estimates better.</param>
    /// <param name="maxIterations">Iteration cap; at least 2 is used.</param>
    /// <param name="seed">Seed for the random starting probes.</param>
    /// <exception cref="ArgumentOutOfRangeException">The operator's order is negative, or the n x t probe panel would not fit a buffer.</exception>
    /// <exception cref="AllocationLimitException">A probe panel would exceed <see cref="TensileLimits.MaxElements"/>.</exception>
    public static NormEstimateResult Of(
        IAdjointOperator<Complex> op,
        int columns = DefaultColumns,
        int maxIterations = DefaultMaxIterations,
        int seed = DefaultSeed) =>
        Estimate<Complex, ComplexKernels>(op, columns, maxIterations, seed);

    /// <summary>
    /// The algorithm, once over the element type, for the generic algorithms
    /// that need an estimate without knowing <c>T</c>. Whether sign vectors are
    /// +/-1 -- which is what makes the tests for parallel columns meaningful --
    /// comes from <c>TKernels.SignsAreDiscrete</c>. For <see cref="double"/>
    /// this performs the same operations, in the same order and with the same
    /// random draws, as the real-only estimator it replaced, so real estimates
    /// did not move.
    /// </summary>
    internal static NormEstimateResult Estimate<T, TKernels>(
        IAdjointOperator<T> op,
        int columns = DefaultColumns,
        int maxIterations = DefaultMaxIterations,
        int seed = DefaultSeed)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
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

        T[] x = Storage.Array<T>(extent, panelPurpose);
        T[] y = Storage.Array<T>(extent, panelPurpose);
        T[] s = Storage.Array<T>(extent, panelPurpose);
        T[] sOld = Storage.Array<T>(extent, panelPurpose);
        T[] z = Storage.Array<T>(extent, panelPurpose);
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

            op.Apply(ReadOnlyMatrixView<T>.Bind(x, panel), MatrixView<T>.Bind(y, panel));
            products += t;

            int argmax = 0;
            est = ColumnOneNorm<T, TKernels>(y, 0, n);

            for (int j = 1; j < t; j++)
            {
                double value = ColumnOneNorm<T, TKernels>(y, j * n, n);
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

            // S := sign(Y), taking sign(0) as 1. The previous S is kept to
            // detect a repeat, which is the other termination test -- for real
            // signs only, since only a discrete set of signs can repeat.
            (s, sOld) = (sOld, s);
            for (int i = 0; i < extent; i++) s[i] = TKernels.Sign(y[i]);

            if (TKernels.SignsAreDiscrete && t > 1)
            {
                if (AllColumnsParallel(n, t, s, sOld)) break;
                MakeColumnsDistinct(rng, n, t, s, sOld);
            }

            op.ApplyAdjoint(ReadOnlyMatrixView<T>.Bind(s, panel), MatrixView<T>.Bind(z, panel));
            products += t;

            for (int i = 0; i < n; i++)
            {
                double best = 0.0;
                for (int j = 0; j < t; j++)
                    best = Math.Max(best, TKernels.Magnitude(z[j * n + i]));
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
                x[j * n + index] = T.One;
            }
        }

        return new NormEstimateResult(est, iterations, products);
    }

    /// <summary>
    /// Starting probe: one column of ones, the rest random +/-1 and distinct
    /// from their predecessors, all scaled to unit 1-norm. Real in both the
    /// real and the complex estimator.
    /// </summary>
    private static void InitialProbe<T>(Random rng, int n, int t, T[] x)
        where T : unmanaged, INumberBase<T>
    {
        for (int i = 0; i < n; i++) x[i] = T.One;

        for (int j = 1; j < t; j++)
        {
            for (int attempt = 0; attempt < ResampleLimit; attempt++)
            {
                FillRandomSigns(rng, x, j * n, n);
                if (!IsParallelToAny(n, x, j * n, x, j)) break;
            }
        }

        // +/-1 times the rounded 1/n is exactly the rounded +/-1/n, so this is
        // the same panel as dividing each entry by n.
        T scale = T.CreateTruncating(1.0 / n);
        for (int i = 0; i < n * t; i++) x[i] *= scale;
    }

    private static void FillRandomSigns<T>(Random rng, T[] panel, int start, int n)
        where T : unmanaged, INumberBase<T>
    {
        for (int i = 0; i < n; i++) panel[start + i] = rng.Next(2) == 0 ? -T.One : T.One;
    }

    /// <summary>
    /// Two +/-1 vectors are parallel exactly when one is the other or its
    /// negation, entry by entry. Exact comparisons on exact values; the same
    /// verdict as |u.v| = n, without the arithmetic.
    /// </summary>
    private static bool IsParallel<T>(ReadOnlySpan<T> u, ReadOnlySpan<T> v)
        where T : unmanaged, INumberBase<T>
    {
        bool same = true, opposite = true;

        for (int i = 0; i < u.Length; i++)
        {
            if (u[i] != v[i]) same = false;
            if (u[i] != -v[i]) opposite = false;
            if (!same && !opposite) return false;
        }

        return true;
    }

    /// <summary>Whether the column at <paramref name="start"/> is parallel to any of the first <paramref name="count"/> columns of <paramref name="panel"/>.</summary>
    private static bool IsParallelToAny<T>(int n, T[] column, int start, T[] panel, int count)
        where T : unmanaged, INumberBase<T>
    {
        ReadOnlySpan<T> candidate = column.AsSpan(start, n);

        for (int j = 0; j < count; j++)
        {
            if (IsParallel<T>(candidate, panel.AsSpan(j * n, n))) return true;
        }

        return false;
    }

    /// <summary>Every column of S has already appeared in S_old.</summary>
    private static bool AllColumnsParallel<T>(int n, int t, T[] s, T[] sOld)
        where T : unmanaged, INumberBase<T>
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
    private static void MakeColumnsDistinct<T>(Random rng, int n, int t, T[] s, T[] sOld)
        where T : unmanaged, INumberBase<T>
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

    private static double ColumnOneNorm<T, TKernels>(T[] panel, int start, int n)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += TKernels.Magnitude(panel[start + i]);
        return sum;
    }
}

/// <summary>
/// Condition estimation in the 1-norm, the equivalent of LAPACK's dgecon and
/// zgecon. Reached through the <c>ReciprocalCondition</c> extensions, which
/// supply the norm of the original matrix themselves.
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
    /// reason <see cref="LuDecomposition{T}.PivotRatio"/> is not a substitute.
    /// </summary>
    public static double ReciprocalOne(
        double normOfA,
        LuDecomposition<double> lu,
        int columns = NormEstimate.DefaultColumns,
        int maxIterations = NormEstimate.DefaultMaxIterations,
        int seed = NormEstimate.DefaultSeed)
    {
        ArgumentNullException.ThrowIfNull(lu);

        if (lu.IsSingular || normOfA == 0.0 || lu.Rows == 0) return 0.0;

        double inverseNorm = NormEstimate.Of(new LuInverseOperator<double>(lu), columns, maxIterations, seed).Value;

        return Reciprocal(normOfA, inverseNorm);
    }

    /// <summary>The same estimate for a complex factorization: the inverse's adjoint is the adjoint solve.</summary>
    public static double ReciprocalOne(
        double normOfA,
        LuDecomposition<Complex> lu,
        int columns = NormEstimate.DefaultColumns,
        int maxIterations = NormEstimate.DefaultMaxIterations,
        int seed = NormEstimate.DefaultSeed)
    {
        ArgumentNullException.ThrowIfNull(lu);

        if (lu.IsSingular || normOfA == 0.0 || lu.Rows == 0) return 0.0;

        double inverseNorm = NormEstimate.Of(new LuInverseOperator<Complex>(lu), columns, maxIterations, seed).Value;

        return Reciprocal(normOfA, inverseNorm);
    }

    private static double Reciprocal(double normOfA, double inverseNorm) =>
        inverseNorm == 0.0 ? 0.0 : 1.0 / (normOfA * inverseNorm);
}
