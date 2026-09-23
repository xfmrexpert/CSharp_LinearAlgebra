namespace Tensile;

/// <summary>
/// The action of the matrix exponential, exp(tA)B, computed without ever
/// forming exp(tA). After Al-Mohy and Higham, "Computing the Action of the
/// Matrix Exponential, with an Application to Exponential Integrators",
/// SIAM J. Sci. Comput. 33(2):488-511, 2011.
///
/// This is the algorithm that matters for a transient sweep. <see cref="MatrixExponential.Expm(Matrix{double}, Workspace)"/>
/// costs on the order of twenty products of order n and then you still have to
/// multiply by B; this evaluates a truncated Taylor series of exp(tA/s)
/// applied to B, s times, so the cost is a few dozen applications of A to a
/// panel. When B is a single vector each of those is O(n^2) rather than
/// O(n^3), which is the whole point.
///
/// Two things make it more than a naive Taylor series:
///
/// - The scaling parameter s and the truncation degree m are chosen together
///   to minimise m*s, the number of applications, subject to a backward error
///   bound. That choice uses estimates of ||A^p||^(1/p) rather than ||A||,
///   for the same reason the 2009 exponential does: for a nonnormal matrix the
///   former can be very much smaller, and using ||A||
///   would overscale badly.
/// - The inner loop stops early when the remaining terms cannot change the
///   result at the working precision, so the degree m is a cap rather than a
///   count.
///
/// The operator overload takes <see cref="ILinearOperator"/> and not a matrix,
/// which is the point of that interface: an FEM or MTL operator that applies A
/// without assembling it plugs in here directly, and nothing in this algorithm
/// ever needs A's entries or its transpose.
/// </summary>
public static class MatrixExponentialAction
{
    /// <summary>
    /// The largest ||tA|| at which a degree-m truncated Taylor series meets
    /// the backward error bound at double precision, from Table 3.1 of the
    /// paper. Degrees 1 to 30, then every fifth to 55.
    ///
    /// Transcribed, and not derivable: like the Padé thresholds in
    /// <see cref="MatrixExponential"/>, these come from a high-precision
    /// evaluation of a backward error function. What guards them is the
    /// accuracy suite, which compares against <see cref="MatrixExponential.Expm(Matrix{double}, Workspace)"/>
    /// — an implementation that is itself independently verified, so this
    /// algorithm gets a far better oracle than that one had.
    /// </summary>
    private static readonly int[] Degrees =
    [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        35, 40, 45, 50, 55
    ];

    /// <inheritdoc cref="Degrees"/>
    private static readonly double[] Thetas =
    [
        2.29e-16, 2.58e-8, 1.39e-5, 3.40e-4, 2.40e-3,
        9.07e-3, 2.38e-2, 5.00e-2, 8.96e-2, 1.44e-1,
        2.14e-1, 3.00e-1, 4.00e-1, 5.14e-1, 6.41e-1,
        7.81e-1, 9.31e-1, 1.09, 1.26, 1.44,
        1.62, 1.82, 2.01, 2.22, 2.43,
        2.64, 2.86, 3.08, 3.31, 3.54,
        4.7, 6.0, 7.2, 8.5, 9.9
    ];

    /// <summary>The largest degree tabulated, and so the largest the search may pick.</summary>
    private const int MaxDegree = 55;

    /// <summary>
    /// The largest power the sharp bound uses, from the paper's p_max rule:
    /// the largest p with p(p-1) &lt;= m_max + 1, searched near sqrt(m_max).
    /// For m_max = 55 that is 8, since 8*7 = 56 fits and 9*8 = 72 does not.
    /// </summary>
    private const int MaxPower = 8;

    /// <summary>Probe columns for the ell parameter of the paper's bounds.</summary>
    private const int Ell = 2;

    /// <summary>Unit roundoff for IEEE binary64, and the early-termination tolerance.</summary>
    private const double UnitRoundoff = 1.1102230246251565e-16;

    /// <summary>
    /// exp(tA)B for a dense A, without forming the exponential.
    ///
    /// The dense path gets two things the matrix-free one cannot: the trace
    /// shift, which replaces A by A - mu*I and undoes it with a scalar factor
    /// (this costs nothing and can reduce the norm a great deal, which reduces
    /// s), and sharp ||A^p||^(1/p) estimates, which need products with A
    /// transposed.
    /// </summary>
    /// <param name="a">The operator. Must be square. Not modified.</param>
    /// <param name="b">The panel to apply exp(tA) to, with as many rows as A. Not modified.</param>
    /// <param name="t">The time, or any scalar multiplier on A. May be negative or zero.</param>
    /// <returns>A new matrix holding exp(tA)B, the same shape as B.</returns>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    /// <exception cref="ArgumentException">A is not square, or B has the wrong number of rows.</exception>
    /// <remarks>
    /// There is deliberately no <see cref="Workspace"/> parameter. Every
    /// application of A here goes through the unpacked panel path, which has
    /// no packing buffers and no kernel choice to configure, so a workspace
    /// would have nothing to do and taking one would imply otherwise. A wide
    /// B would be better served by GEMM, and that would reintroduce the
    /// parameter -- see the open item in CLAUDE.md, which is unmeasured.
    /// </remarks>
    public static Matrix<double> Expmv(
        this Matrix<double> a, ReadOnlyMatrixView<double> b, double t = 1.0) =>
        Expmv(a, b, t, diagnostics: null);

    /// <summary>The dense implementation, with an optional report of the parameters chosen.</summary>
    internal static Matrix<double> Expmv(
        Matrix<double> a,
        ReadOnlyMatrixView<double> b,
        double t,
        ExpmvDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"The exponential action requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        if (b.Rows != a.Rows)
            throw new ArgumentException($"B has {b.Rows} rows, expected the operator's order {a.Rows}.", nameof(b));

        int n = a.Rows;
        if (n == 0 || b.Columns == 0) return Matrix.From(b);

        // The trace shift. exp(tA) = exp(t*mu) * exp(t*(A - mu*I)), and the
        // shifted matrix usually has a much smaller norm, which directly
        // reduces s. The scalar is folded back in one factor per outer step.
        double mu = Trace(a) / n;
        var shifted = ShiftDiagonal(a, -mu);

        double oneNorm = shifted.OneNorm();

        // The sharp bound needs ||A^p||^(1/p), which the estimator reaches
        // through products with A and A^T -- available here because a dense
        // matrix is transposable, and the reason the matrix-free overload
        // cannot use it.
        double Alpha(int p) => Math.Max(DPower(shifted, p), DPower(shifted, p + 1));

        return Run(new DenseMatrixOperator(shifted), b, t, mu, oneNorm, Alpha, diagnostics);
    }

    /// <summary>
    /// exp(tA)B for an operator that is applied rather than stored.
    ///
    /// Only <see cref="ILinearOperator.Apply"/> is used: no transpose, no
    /// entries, no trace. That is why this takes the narrow interface, and it
    /// is the form a matrix-free FEM or MTL operator plugs into.
    ///
    /// The price is the parameter choice. Without products by A^T the
    /// ||A^p||^(1/p) quantities cannot be estimated, so the scaling falls back
    /// to <paramref name="oneNormBound"/>. Since ||A^p||^(1/p) is never larger
    /// than ||A||, that is always safe -- it can only choose a larger s than
    /// necessary, never a smaller one -- but for a strongly nonnormal operator
    /// it can be much more work than the dense path would do. Supply the
    /// tightest bound you have.
    /// </summary>
    /// <param name="op">The operator. Applied, never inspected.</param>
    /// <param name="b">The panel to apply exp(tA) to, with as many rows as the operator's order.</param>
    /// <param name="t">The time, or any scalar multiplier on A. May be negative or zero.</param>
    /// <param name="oneNormBound">An upper bound on ||A||_1. Must be finite and not negative.</param>
    /// <returns>A new matrix holding exp(tA)B, the same shape as B.</returns>
    /// <exception cref="ArgumentNullException">The operator is null.</exception>
    /// <exception cref="ArgumentException">B has the wrong number of rows.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The norm bound is negative or not finite.</exception>
    public static Matrix<double> Expmv(
        ILinearOperator op, ReadOnlyMatrixView<double> b, double t, double oneNormBound) =>
        Expmv(op, b, t, oneNormBound, diagnostics: null);

    /// <summary>The matrix-free implementation, with an optional report of the parameters chosen.</summary>
    internal static Matrix<double> Expmv(
        ILinearOperator op,
        ReadOnlyMatrixView<double> b,
        double t,
        double oneNormBound,
        ExpmvDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(op);

        if (b.Rows != op.Order)
            throw new ArgumentException($"B has {b.Rows} rows, expected the operator's order {op.Order}.", nameof(b));

        if (!double.IsFinite(oneNormBound) || oneNormBound < 0.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(oneNormBound), oneNormBound, "The norm bound must be finite and not negative.");
        }

        if (op.Order == 0 || b.Columns == 0) return Matrix.From(b);

        // No trace, so no shift: mu = 0. And no transpose, so no alpha.
        return Run(op, b, t, mu: 0.0, oneNormBound, alpha: null, diagnostics);
    }

    /// <summary>
    /// The algorithm proper, shared by both overloads: choose the parameters,
    /// then run the scaled Taylor recurrence.
    /// </summary>
    private static Matrix<double> Run(
        ILinearOperator op,
        ReadOnlyMatrixView<double> b,
        double t,
        double mu,
        double oneNorm,
        Func<int, double>? alpha,
        ExpmvDiagnostics? diagnostics)
    {
        int n = op.Order;
        int columns = b.Columns;

        // Everything in the parameter selection is a norm of tA, so it scales
        // by |t|. The reference implementation multiplies by t rather than
        // |t|, which gives a negative "norm" and a non-positive s for t < 0;
        // the magnitude is what the bounds mean, and it makes exp(-tA)B work.
        double scale = Math.Abs(t);
        double scaledNorm = scale * oneNorm;

        int m;
        int s;

        if (scaledNorm == 0.0)
        {
            // A is zero, or t is. exp(tA) is the identity and the only thing
            // left is the shift's scalar.
            m = 0;
            s = 1;
        }
        else
        {
            (m, s) = SelectParameters(scaledNorm, alpha, scale, columns);
        }

        Report(diagnostics, m, s);

        var f = Matrix.From(b);
        var work = Matrix.From(b);
        var next = new Matrix<double>(n, columns);

        double eta = Math.Exp(t * mu / s);

        for (int i = 0; i < s; i++)
        {
            double c1 = work.InfinityNorm();

            for (int j = 0; j < m; j++)
            {
                double coefficient = t / (s * (double)(j + 1));

                op.Apply(work.ReadOnlyView, next.View);
                ScaleInto(next, coefficient, work);

                double c2 = work.InfinityNorm();
                AddInto(f, work);

                if (diagnostics is not null) diagnostics.Applications++;

                // The remaining terms cannot move the result at this
                // precision, so the degree is a cap rather than a count. This
                // is where most of the saving against a fixed-degree series
                // comes from.
                if (c1 + c2 <= UnitRoundoff * f.InfinityNorm())
                {
                    if (diagnostics is not null) diagnostics.EarlyExits++;
                    break;
                }

                c1 = c2;
            }

            ScaleInPlace(f, eta);
            CopyInto(f, work);
        }

        return f;
    }

    /// <summary>
    /// Choose the degree m and the scaling s together, minimising m*s -- the
    /// number of operator applications -- subject to the backward error bound.
    ///
    /// Two regimes, as in the paper. For a small enough norm the crude bound
    /// from ||tA|| is already within a constant of the sharp one, and using it
    /// avoids the estimator entirely (condition 3.13). Otherwise the sharp
    /// bound over ||A^p||^(1/p) is worth its cost -- when it is available at
    /// all, which for a matrix-free operator it is not.
    ///
    /// The search runs in <c>double</c> throughout and narrows to an int only
    /// once. That is not fastidiousness: theta_1 is 2.29e-16, so s for degree 1
    /// is about 4e15 times the norm, which overflows an int for any input at
    /// all. The reference implementation is in Python, where that value is an
    /// arbitrary-precision integer that simply loses the minimisation; here it
    /// would trap, since this assembly is compiled with
    /// <c>CheckForOverflowUnderflow</c>. Degree 1 never wins, but the
    /// arithmetic still has to be able to evaluate its cost in order to
    /// discard it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The chosen parameters would need more operator applications than can be
    /// counted, which means ||tA|| is far too large for this method. Finding 11
    /// in reverse: the input is valid and the cost is not.
    /// </exception>
    private static (int Degree, int Scaling) SelectParameters(
        double scaledNorm, Func<int, double>? alpha, double scale, int columns)
    {
        int bestDegree = 0;
        double bestScaling = 0.0;
        double bestCost = double.PositiveInfinity;

        void Consider(int index, double norm)
        {
            double scaling = Math.Ceiling(norm / Thetas[index]);
            double cost = Degrees[index] * scaling;

            if (cost >= bestCost) return;

            bestCost = cost;
            bestDegree = Degrees[index];
            bestScaling = scaling;
        }

        if (alpha is null || SatisfiesCondition313(scaledNorm, columns))
        {
            for (int index = 0; index < Degrees.Length; index++) Consider(index, scaledNorm);
        }
        else
        {
            // Equation (3.11): for each power p, only degrees from p(p-1)-1 up
            // are admissible, because a lower degree cannot exploit that power.
            for (int p = 2; p <= MaxPower; p++)
            {
                double alphaP = scale * alpha(p);

                for (int index = 0; index < Degrees.Length; index++)
                {
                    if (Degrees[index] < (p * (p - 1)) - 1) continue;
                    Consider(index, alphaP);
                }
            }
        }

        double scalingSteps = Math.Max(bestScaling, 1.0);

        if (!double.IsFinite(bestCost) || scalingSteps > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(scaledNorm),
                scaledNorm,
                $"||tA||_1 of {scaledNorm:E3} would need about {bestCost:E3} operator applications, "
                + "which is not a computation this method can finish. Reduce t, or shift the operator.");
        }

        return (bestDegree, (int)scalingSteps);
    }

    /// <summary>
    /// Condition (3.13): below this norm the crude bound is good enough, and
    /// the estimator's products would cost more than the scaling they save.
    /// </summary>
    private static bool SatisfiesCondition313(double scaledNorm, int columns)
    {
        double a = 2.0 * Ell * MaxPower * (MaxPower + 3);
        double b = Thetas[^1] / (columns * (double)MaxDegree);

        return scaledNorm <= a * b;
    }

    /// <summary>||A^p||_1^(1/p), estimated rather than formed.</summary>
    private static double DPower(Matrix<double> a, int p) =>
        Math.Pow(a.EstimateOneNorm(power: p, columns: Ell), 1.0 / p);

    /// <summary>Record the parameters chosen, when a caller asked for them.</summary>
    private static void Report(ExpmvDiagnostics? diagnostics, int degree, int scaling)
    {
        if (diagnostics is null) return;

        diagnostics.Degree = degree;
        diagnostics.Scaling = scaling;
    }

    private static double Trace(Matrix<double> a)
    {
        double total = 0.0;
        for (int i = 0; i < a.Rows; i++) total += a[i, i];
        return total;
    }

    /// <summary>A copy of <paramref name="a"/> with <paramref name="delta"/> added to its diagonal.</summary>
    private static Matrix<double> ShiftDiagonal(Matrix<double> a, double delta)
    {
        var result = Matrix.From(a.ReadOnlyView);
        for (int i = 0; i < a.Rows; i++) result[i, i] += delta;
        return result;
    }

    /// <summary>target := scale * source, both the same shape.</summary>
    private static void ScaleInto(Matrix<double> source, double scale, Matrix<double> target)
    {
        for (int j = 0; j < source.Columns; j++)
        {
            ReadOnlySpan<double> from = source.ReadOnlyView.Column(j);
            Span<double> to = target.Column(j);
            for (int i = 0; i < from.Length; i++) to[i] = from[i] * scale;
        }
    }

    private static void ScaleInPlace(Matrix<double> a, double scale)
    {
        if (scale == 1.0) return;

        for (int j = 0; j < a.Columns; j++)
        {
            Span<double> column = a.Column(j);
            for (int i = 0; i < column.Length; i++) column[i] *= scale;
        }
    }

    /// <summary>target := target + source.</summary>
    private static void AddInto(Matrix<double> target, Matrix<double> source)
    {
        for (int j = 0; j < target.Columns; j++)
        {
            ReadOnlySpan<double> from = source.ReadOnlyView.Column(j);
            Span<double> to = target.Column(j);
            for (int i = 0; i < to.Length; i++) to[i] += from[i];
        }
    }

    /// <summary>target := source.</summary>
    private static void CopyInto(Matrix<double> source, Matrix<double> target)
    {
        for (int j = 0; j < source.Columns; j++)
            source.ReadOnlyView.Column(j).CopyTo(target.Column(j));
    }
}

/// <summary>
/// The parameters <see cref="MatrixExponentialAction"/> chose, and what the
/// recurrence then did.
///
/// Internal for the same reason <c>ExpmDiagnostics</c> is: the choice is an
/// implementation detail, observable only so the suite can check that the
/// parameter selection is doing its job rather than that the answer happened
/// to come out right. A degree and scaling that were always the same would
/// mean the selection had collapsed, and no accuracy test would notice.
/// </summary>
internal sealed class ExpmvDiagnostics
{
    /// <summary>The Taylor degree chosen, as a cap. Zero when tA is zero.</summary>
    public int Degree { get; set; }

    /// <summary>How many scaling steps the recurrence took. At least 1.</summary>
    public int Scaling { get; set; }

    /// <summary>How many times the operator was actually applied.</summary>
    public int Applications { get; set; }

    /// <summary>How many inner loops stopped early because the terms had stopped mattering.</summary>
    public int EarlyExits { get; set; }
}
