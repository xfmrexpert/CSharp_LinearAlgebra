using System.Runtime.InteropServices;

namespace Tensile.Primitives;

/// <summary>Outcome of a 1-norm estimate.</summary>
/// <param name="Value">The estimate, always a lower bound on the true 1-norm.</param>
/// <param name="Iterations">Iterations performed.</param>
/// <param name="Products">Operator applications, counting each probe column.</param>
public readonly record struct NormEstimateResult(double Value, int Iterations, int Products);

/// <summary>
/// Matrix 1-norms, exact and estimated.
/// </summary>
public static unsafe class Norms
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
/// </summary>
public static unsafe class NormEstimate
{
    /// <summary>Probe columns. Higham and Tisseur recommend 2; more costs more products.</summary>
    public const int DefaultColumns = 2;

    /// <summary>Iteration cap. MATLAB's normest1 uses 5; convergence is usually in 2 or 3.</summary>
    public const int DefaultMaxIterations = 5;

    /// <summary>Fixed so that estimates are reproducible run to run.</summary>
    public const int DefaultSeed = 20260915;

    /// <summary>Attempts before accepting a probe column parallel to an earlier one.</summary>
    private const int ResampleLimit = 8;

    /// <summary>
    /// Estimate ||A||_1 for the dense n x n matrix A, or ||A^power||_1 without
    /// forming the power.
    /// </summary>
    public static NormEstimateResult OfMatrix(
        int n, double* a, int lda,
        int power = 1,
        int columns = DefaultColumns,
        int maxIterations = DefaultMaxIterations,
        int seed = DefaultSeed)
    {
        using var op = new DenseMatrixOperator(n, a, lda, power);
        return Of(op, columns, maxIterations, seed);
    }

    /// <summary>Estimate ||A||_1 for an arbitrary square operator.</summary>
    public static NormEstimateResult Of(
        ILinearOperator op,
        int columns = DefaultColumns,
        int maxIterations = DefaultMaxIterations,
        int seed = DefaultSeed)
    {
        ArgumentNullException.ThrowIfNull(op);

        int n = op.Order;
        if (n == 0) return new NormEstimateResult(0.0, 0, 0);

        int t = Math.Clamp(columns, 1, n);

        // Higham and Tisseur's convergence tests compare against the previous
        // iteration, so fewer than two iterations cannot terminate meaningfully.
        int itmax = Math.Max(maxIterations, 2);

        // n * t is the probe panel. Computed in long and checked: with an
        // operator free to report any Order, n = t = 65536 wraps the int
        // product to exactly zero, a zero-length buffer comes back, and the
        // estimator then writes 2^32 doubles into it. Rejecting here is I2/I7.
        long panelExtent = (long)n * t;

        if (panelExtent > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(columns),
                $"An order-{n} operator with {t} probe columns needs a {panelExtent}-element panel, "
                + $"which exceeds the {int.MaxValue} a buffer can address.");
        }

        int panel = (int)panelExtent;

        double* x = Alloc(panel);
        double* y = Alloc(panel);
        double* s = Alloc(panel);
        double* sOld = Alloc(panel);
        double* z = Alloc(panel);
        double* h = Alloc(n);

        try
        {
            var rng = new Random(seed);
            var order = new int[n];
            var scratchOrder = new int[n];
            var keys = new double[n];
            var used = new bool[n];
            var current = new int[t];

            InitialProbe(rng, n, t, x);
            new Span<double>(s, panel).Clear();

            double est = 0.0;
            double estOld = 0.0;
            int bestIndex = -1;
            int iterations = 0;
            int products = 0;

            for (int k = 1; ; k++)
            {
                iterations = k;

                op.Apply(t, x, n, y, n);
                products += t;

                int argmax = 0;
                est = ColumnOneNorm(n, y);

                for (int j = 1; j < t; j++)
                {
                    double value = ColumnOneNorm(n, y + (nint)j * n);
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
                double* previousSigns = sOld;
                sOld = s;
                s = previousSigns;
                for (int i = 0; i < panel; i++) s[i] = y[i] >= 0.0 ? 1.0 : -1.0;

                if (t > 1)
                {
                    if (AllColumnsParallel(n, t, s, sOld)) break;
                    MakeColumnsDistinct(rng, n, t, s, sOld);
                }

                op.ApplyTranspose(t, s, n, z, n);
                products += t;

                for (int i = 0; i < n; i++)
                {
                    double best = 0.0;
                    for (int j = 0; j < t; j++)
                        best = Math.Max(best, Math.Abs(z[(nint)j * n + i]));
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

                new Span<double>(x, panel).Clear();

                for (int j = 0; j < t; j++)
                {
                    int index = order[j];
                    current[j] = index;
                    used[index] = true;
                    x[(nint)j * n + index] = 1.0;
                }
            }

            return new NormEstimateResult(est, iterations, products);
        }
        finally
        {
            NativeMemory.AlignedFree(x);
            NativeMemory.AlignedFree(y);
            NativeMemory.AlignedFree(s);
            NativeMemory.AlignedFree(sOld);
            NativeMemory.AlignedFree(z);
            NativeMemory.AlignedFree(h);
        }
    }

    /// <summary>
    /// Starting probe: one column of ones, the rest random +/-1 and distinct
    /// from their predecessors, all scaled to unit 1-norm.
    /// </summary>
    private static void InitialProbe(Random rng, int n, int t, double* x)
    {
        for (int i = 0; i < n; i++) x[i] = 1.0;

        for (int j = 1; j < t; j++)
        {
            double* column = x + (nint)j * n;

            for (int attempt = 0; attempt < ResampleLimit; attempt++)
            {
                FillRandomSigns(rng, n, column);
                if (!IsParallelToAny(n, column, x, j)) break;
            }
        }

        for (int i = 0; i < n * t; i++) x[i] /= n;
    }

    private static void FillRandomSigns(Random rng, int n, double* column)
    {
        for (int i = 0; i < n; i++) column[i] = rng.Next(2) == 0 ? -1.0 : 1.0;
    }

    /// <summary>
    /// Two +/-1 vectors are parallel exactly when |x.y| = n. The entries are
    /// integers well inside the exactly representable range, so this comparison
    /// is safe.
    /// </summary>
    private static bool IsParallel(int n, double* u, double* v) =>
        Math.Abs(Blas1.Dot(n, u, v)) == n;

    private static bool IsParallelToAny(int n, double* column, double* panel, int count)
    {
        for (int j = 0; j < count; j++)
        {
            if (IsParallel(n, column, panel + (nint)j * n)) return true;
        }

        return false;
    }

    /// <summary>Every column of S has already appeared in S_old.</summary>
    private static bool AllColumnsParallel(int n, int t, double* s, double* sOld)
    {
        for (int j = 0; j < t; j++)
        {
            if (!IsParallelToAny(n, s + (nint)j * n, sOld, t)) return false;
        }

        return true;
    }

    /// <summary>
    /// Resample any column of S that duplicates an earlier column of S or a
    /// column of S_old. Duplicates waste a product without adding information.
    /// </summary>
    private static void MakeColumnsDistinct(Random rng, int n, int t, double* s, double* sOld)
    {
        for (int j = 0; j < t; j++)
        {
            double* column = s + (nint)j * n;

            for (int attempt = 0; attempt < ResampleLimit; attempt++)
            {
                if (!IsParallelToAny(n, column, s, j) && !IsParallelToAny(n, column, sOld, t))
                    break;

                FillRandomSigns(rng, n, column);
            }
        }
    }

    private static void SortIndicesDescending(int n, double* h, double[] keys, int[] order)
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

    private static double ColumnOneNorm(int n, double* column)
    {
        double sum = 0.0;
        for (int i = 0; i < n; i++) sum += Math.Abs(column[i]);
        return sum;
    }

    private static double* Alloc(int count)
    {
        // A negative count would sign-extend to an absurd nuint and be
        // reported as an allocation failure, which is the wrong diagnosis for
        // an argument error. Every caller has already validated, so this is a
        // guard against the next caller that does not.
        ArgumentOutOfRangeException.ThrowIfNegative(count);

        return (double*)NativeMemory.AlignedAlloc((nuint)count * sizeof(double), 64);
    }
}

/// <summary>
/// Condition estimation in the 1-norm, the equivalent of LAPACK's dgecon.
/// </summary>
public static unsafe class Condition
{
    /// <summary>
    /// Estimate 1/cond_1(A) = 1/(||A||_1 * ||A^-1||_1) from an LU
    /// factorization.
    ///
    /// <paramref name="normOfA"/> is ||A||_1 of the ORIGINAL matrix and must be
    /// computed before factoring, since the factorization overwrites A. dgecon
    /// takes the same argument for the same reason.
    ///
    /// Returns 0 for an exactly singular factorization. The estimator
    /// underestimates ||A^-1||_1, so the value returned is an OVER-estimate of
    /// the reciprocal condition number: a small result reliably means
    /// ill-conditioning, a large one is weaker evidence of good conditioning.
    /// This is the same asymmetry dgecon has, and the reason
    /// <see cref="LuFactorization.PivotRatio"/> is not a substitute.
    /// </summary>
    public static double ReciprocalOne(
        double normOfA,
        LuFactorization lu,
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
