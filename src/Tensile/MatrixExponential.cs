using System.Numerics;

namespace Tensile;

/// <summary>
/// The matrix exponential, by scaling and squaring with Padé approximants,
/// after Al-Mohy and Higham, "A New Scaling and Squaring Algorithm for the
/// Matrix Exponential", SIAM J. Matrix Anal. Appl. 31(3):970-989, 2009.
///
/// This is the 2009 algorithm, not Higham's 2005 one. The difference is how
/// the scaling parameter is chosen: 2005 works from ||A||, 2009 from estimates
/// of ||A^k||^(1/k), which for a nonnormal matrix can be very much smaller.
/// The consequence is less overscaling, and overscaling is exactly the failure
/// mode that matters for stiff problems -- every squaring step is another
/// chance to amplify rounding error, so a scaling parameter one larger than
/// necessary costs accuracy as well as a product.
///
/// The ||A^k||^(1/k) quantities come from <see cref="NormEstimate"/> applied
/// to <see cref="DenseMatrixOperator"/> raised to a power, which never forms
/// A^k: the estimator needs a few products with A and A^T, so estimating
/// ||A^10|| costs O(n^2) per probe rather than the O(n^3) of forming the
/// power. That is what the operator's <c>power</c> parameter is for.
///
/// Accuracy is backward-stable in the sense the paper establishes: the
/// computed result is the exact exponential of A + E with ||E|| bounded
/// relative to ||A|| at the unit roundoff. It is NOT forward-accurate for
/// every matrix, and no algorithm for the exponential is -- see Moler and Van
/// Loan, "Nineteen Dubious Ways to Compute the Exponential of a Matrix".
/// A matrix whose exponential is genuinely ill-conditioned will lose digits
/// here as it would anywhere.
///
/// The algorithm is written once, over the element type, and runs natively
/// for <see cref="double"/> and <see cref="Complex"/>: products, norms, the
/// estimator, the LU of the Padé denominator and the <c>ell</c> test all come
/// through <see cref="IElementKernels{T}"/>. For double it performs the same
/// operations as the real-only implementation it replaced.
/// </summary>
public static partial class MatrixExponential
{
    /// <summary>
    /// The largest ||A|| at which each Padé degree achieves backward error
    /// within the unit roundoff, from Table 3.1 of the paper. Transcribed,
    /// and not independently derivable -- they come from a high-precision
    /// evaluation of the backward error function, so unlike the Padé and
    /// <c>ell</c> coefficients below there is no cheap way for a test to
    /// re-derive them. What guards them is the accuracy suite, which compares
    /// against an independent Taylor oracle across matrix classes that select
    /// every one of these branches.
    /// </summary>
    private const double Theta3 = 1.495585217958292e-2;

    /// <inheritdoc cref="Theta3"/>
    private const double Theta5 = 2.539398330063230e-1;

    /// <inheritdoc cref="Theta3"/>
    private const double Theta7 = 9.504178996162932e-1;

    /// <inheritdoc cref="Theta3"/>
    private const double Theta9 = 2.097847961257068e0;

    /// <summary>
    /// The threshold the scaling parameter is chosen against: s is the
    /// smallest value with 2^-s * eta_5 &lt;= this.
    ///
    /// Note this is 4.25 and NOT the 5.371920351148152 that Table 3.1 lists as
    /// theta_13. The table value is the backward error threshold for the
    /// degree-13 approximant; Algorithm 3.1 deliberately scales against a
    /// smaller number, and the two are easy to confuse because both are called
    /// theta_13 in the literature. Using the table value here would underscale
    /// by one step on some inputs.
    /// </summary>
    private const double ScalingThreshold = 4.25;

    /// <summary>Unit roundoff for IEEE binary64, 2^-53.</summary>
    private const double UnitRoundoff = 1.1102230246251565e-16;

    /// <summary>
    /// Padé numerator coefficients, b[j] multiplying A^j, for each degree.
    ///
    /// Integers, and exactly representable despite most of them exceeding
    /// 2^53: each carries enough factors of two to fit a 53-bit significand.
    /// The largest, b[0] for degree 13, is 26!/13! = 64764752532480000 =
    /// 7905853580625 * 2^13. The suite pins that separately rather than
    /// leaving it as an assertion in a comment.
    ///
    /// Writing them as integers is also what makes them checkable: the tests
    /// re-derive every entry from b[j] = (2m-j)! / (j! (m-j)!), the normalised
    /// form of the Padé coefficient (2m-j)! m! / ((2m)! j! (m-j)!). That turns
    /// a transcription error from a silent accuracy loss into a failure.
    /// </summary>
    private static readonly double[] Pade3 = [120, 60, 12, 1];

    /// <inheritdoc cref="Pade3"/>
    private static readonly double[] Pade5 = [30240, 15120, 3360, 420, 30, 1];

    /// <inheritdoc cref="Pade3"/>
    private static readonly double[] Pade7 =
        [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1];

    /// <inheritdoc cref="Pade3"/>
    private static readonly double[] Pade9 =
        [17643225600, 8821612800, 2075673600, 302702400, 30270240, 2162160, 110880, 3960, 90, 1];

    /// <inheritdoc cref="Pade3"/>
    private static readonly double[] Pade13 =
    [
        64764752532480000, 32382376266240000, 7771770303897600, 1187353796428800,
        129060195264000, 10559470521600, 670442572800, 33522128640,
        1323241920, 40840800, 960960, 16380, 182, 1
    ];

    /// <summary>
    /// exp(A), the matrix exponential -- the sum of A^k/k!, NOT the
    /// element-wise exponential of A's entries.
    ///
    /// The name is the one the numerical literature uses, and is deliberately
    /// not <c>Exp</c>, which would be indistinguishable from an element-wise
    /// map at the call site.
    ///
    /// Cost is dominated by matrix products: three to form the powers, two
    /// more to evaluate the degree-13 approximant, one LU factorization and
    /// solve, and one product per squaring step. For a well-scaled matrix
    /// expect on the order of 15 to 25 products of order n.
    /// </summary>
    /// <param name="a">The matrix to exponentiate. Must be square. Not modified.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <returns>A new matrix holding exp(A).</returns>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    /// <exception cref="ArgumentException">The matrix is not square.</exception>
    /// <exception cref="InvalidOperationException">
    /// The Padé denominator was exactly singular. This cannot happen in exact
    /// arithmetic -- the denominator's eigenvalues are q_m evaluated at A's
    /// eigenvalues, and q_m has no zeros in the region the scaling confines
    /// them to -- so it indicates an input that is not finite.
    /// </exception>
    public static Matrix<double> Expm(this Matrix<double> a, Workspace? workspace = null) =>
        Expm(a, workspace, diagnostics: null);

    /// <summary>
    /// The implementation, with an optional report of which branch was taken.
    ///
    /// Internal rather than public because the degree and the squaring count
    /// are an implementation detail a caller should not depend on. The test
    /// suite needs them for a reason finding 10 makes concrete: without a way
    /// to observe the branch, a suite could exercise only the degree-13 path
    /// and leave the other four unverified while still passing.
    /// </summary>
    internal static Matrix<double> Expm(Matrix<double> a, Workspace? workspace, ExpmDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"The matrix exponential requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        // An empty matrix exponentiates to itself, and the 1x1 case is scalar
        // exp -- worth short-circuiting because the machinery below would
        // otherwise form five 1x1 products to reach the same answer.
        if (a.Rows <= 1)
        {
            Report(diagnostics, degree: 0, squarings: 0);
            return Scalar<double, DoubleKernels>(a);
        }

        return ScalingAndSquaring<double, DoubleKernels>(a, workspace ?? Workspace.Shared, diagnostics);
    }

    /// <summary>The empty and 1x1 exponentials, where exp is the scalar one.</summary>
    private static Matrix<T> Scalar<T, TKernels>(Matrix<T> a)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        var result = new Matrix<T>(a.Rows, a.Columns);
        if (a.Rows == 1) result[0, 0] = TKernels.Exp(a[0, 0]);
        return result;
    }

    /// <summary>
    /// Algorithm 3.1 of the paper, for a square matrix of order at least 2,
    /// over any element type with kernels.
    /// </summary>
    private static Matrix<T> ScalingAndSquaring<T, TKernels>(Matrix<T> a, Workspace ws, ExpmDiagnostics? diagnostics)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int n = a.Rows;

        // A^2, A^4 and A^6 are needed by every Padé degree from 7 up and by
        // the degree-3 and degree-5 tests, so they are formed once here rather
        // than per branch. A^8 is formed only if degree 9 is selected.
        var a2 = Product<T, TKernels>(a, a, ws);
        var a4 = Product<T, TKernels>(a2, a2, ws);
        var a6 = Product<T, TKernels>(a2, a4, ws);

        // d4 and d6 are exact, because A^4 and A^6 are in hand and an exact
        // 1-norm of a formed matrix is O(n^2). The paper allows estimates
        // here; exact is strictly safer, since the estimator returns a lower
        // bound and a larger eta only ever selects a more conservative branch.
        double d4 = Math.Pow(TKernels.OneNorm(a4.ReadOnlyView), 1.0 / 4.0);
        double d6 = Math.Pow(TKernels.OneNorm(a6.ReadOnlyView), 1.0 / 6.0);

        double eta1 = Math.Max(d4, d6);
        if (eta1 < Theta3 && Ell<T, TKernels>(a, 3) == 0)
        {
            Report(diagnostics, degree: 3, squarings: 0);
            return EvaluateLowDegree<T, TKernels>(Pade3, a, a2, a4, a6, null, ws);
        }

        if (eta1 < Theta5 && Ell<T, TKernels>(a, 5) == 0)
        {
            Report(diagnostics, degree: 5, squarings: 0);
            return EvaluateLowDegree<T, TKernels>(Pade5, a, a2, a4, a6, null, ws);
        }

        // d8 costs an estimator run rather than a product, and is not needed
        // unless the low degrees have been ruled out.
        double d8 = Math.Pow(EstimatePowerNorm<T, TKernels>(a, 8), 1.0 / 8.0);
        double eta3 = Math.Max(d6, d8);

        if (eta3 < Theta7 && Ell<T, TKernels>(a, 7) == 0)
        {
            Report(diagnostics, degree: 7, squarings: 0);
            return EvaluateLowDegree<T, TKernels>(Pade7, a, a2, a4, a6, null, ws);
        }

        if (eta3 < Theta9 && Ell<T, TKernels>(a, 9) == 0)
        {
            Report(diagnostics, degree: 9, squarings: 0);

            var a8 = Product<T, TKernels>(a4, a4, ws);
            return EvaluateLowDegree<T, TKernels>(Pade9, a, a2, a4, a6, a8, ws);
        }

        double d10 = Math.Pow(EstimatePowerNorm<T, TKernels>(a, 10), 1.0 / 10.0);
        double eta4 = Math.Max(d8, d10);
        double eta5 = Math.Min(eta3, eta4);

        // eta5 == 0 means every power the estimator looked at was zero, i.e. A
        // is nilpotent of low index. log2(0) would be negative infinity, so
        // this is guarded rather than clamped.
        int s = eta5 == 0.0
            ? 0
            : Math.Max((int)Math.Ceiling(Math.Log2(eta5 / ScalingThreshold)), 0);

        // The ell correction is evaluated on the already-scaled matrix, which
        // is why it is added after s rather than folded into the line above.
        var scaled = Scaled<T, TKernels>(a, Math.ScaleB(1.0, -s));
        s += Ell<T, TKernels>(scaled, 13);

        // Rescale from the original rather than from `scaled`, so the ell
        // correction does not compound a rounding error in the first scaling.
        double factor = Math.ScaleB(1.0, -s);
        var b = Scaled<T, TKernels>(a, factor);
        var b2 = Scaled<T, TKernels>(a2, factor * factor);
        var b4 = Scaled<T, TKernels>(a4, Math.Pow(factor, 4));
        var b6 = Scaled<T, TKernels>(a6, Math.Pow(factor, 6));

        Report(diagnostics, degree: 13, squarings: s);

        var x = EvaluateDegree13<T, TKernels>(b, b2, b4, b6, ws);

        // Undo the scaling: exp(A) = (exp(A/2^s))^(2^s), s squarings.
        var square = new Matrix<T>(n, n);
        for (int i = 0; i < s; i++)
        {
            TKernels.Multiply(ws, x.ReadOnlyView, x.ReadOnlyView, square.View, T.One, T.Zero);
            (x, square) = (square, x);
        }

        return x;
    }

    /// <summary>||A^p||_1, estimated through the element type's power operator, never formed.</summary>
    private static double EstimatePowerNorm<T, TKernels>(Matrix<T> a, int power)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T> =>
        NormEstimate.Estimate<T, TKernels>(TKernels.PowerOperator(a, power)).Value;

    /// <summary>
    /// Evaluate r_m for m in {3, 5, 7, 9}, where the numerator splits into
    /// U = A * (odd-indexed terms) and V = (even-indexed terms).
    ///
    /// <paramref name="a8"/> is required for degree 9 and ignored below it.
    /// </summary>
    private static Matrix<T> EvaluateLowDegree<T, TKernels>(
        double[] b,
        Matrix<T> a,
        Matrix<T> a2,
        Matrix<T> a4,
        Matrix<T> a6,
        Matrix<T>? a8,
        Workspace ws)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int m = b.Length - 1;
        int n = a.Rows;

        // W collects the odd terms divided by A: b1*I + b3*A^2 + b5*A^4 + ...
        var w = Identity<T>(n, b[1]);
        AddScaled<T, TKernels>(w, a2, b[3]);
        if (m >= 5) AddScaled<T, TKernels>(w, a4, b[5]);
        if (m >= 7) AddScaled<T, TKernels>(w, a6, b[7]);
        if (m >= 9) AddScaled<T, TKernels>(w, a8!, b[9]);

        var u = new Matrix<T>(n, n);
        TKernels.Multiply(ws, a.ReadOnlyView, w.ReadOnlyView, u.View, T.One, T.Zero);

        var v = Identity<T>(n, b[0]);
        AddScaled<T, TKernels>(v, a2, b[2]);
        if (m >= 5) AddScaled<T, TKernels>(v, a4, b[4]);
        if (m >= 7) AddScaled<T, TKernels>(v, a6, b[6]);
        if (m >= 9) AddScaled<T, TKernels>(v, a8!, b[8]);

        return SolvePade<T, TKernels>(u, v, ws);
    }

    /// <summary>
    /// Evaluate r_13 by the Paterson-Stockmeyer scheme, which reaches degree
    /// 13 in two extra products rather than the six a naive Horner evaluation
    /// would need. That economy is the reason degree 13 is the top of the
    /// ladder rather than something higher.
    /// </summary>
    private static Matrix<T> EvaluateDegree13<T, TKernels>(
        Matrix<T> a, Matrix<T> a2, Matrix<T> a4, Matrix<T> a6, Workspace ws)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        double[] b = Pade13;
        int n = a.Rows;

        // U = A * (A^6 * (b13 A^6 + b11 A^4 + b9 A^2) + b7 A^6 + b5 A^4 + b3 A^2 + b1 I)
        var inner = Scaled<T, TKernels>(a6, b[13]);
        AddScaled<T, TKernels>(inner, a4, b[11]);
        AddScaled<T, TKernels>(inner, a2, b[9]);

        var w = Identity<T>(n, b[1]);
        AddScaled<T, TKernels>(w, a2, b[3]);
        AddScaled<T, TKernels>(w, a4, b[5]);
        AddScaled<T, TKernels>(w, a6, b[7]);

        // beta = 1 accumulates the product onto the terms already in w.
        TKernels.Multiply(ws, a6.ReadOnlyView, inner.ReadOnlyView, w.View, T.One, T.One);

        var u = new Matrix<T>(n, n);
        TKernels.Multiply(ws, a.ReadOnlyView, w.ReadOnlyView, u.View, T.One, T.Zero);

        // V = A^6 * (b12 A^6 + b10 A^4 + b8 A^2) + b6 A^6 + b4 A^4 + b2 A^2 + b0 I
        var innerEven = Scaled<T, TKernels>(a6, b[12]);
        AddScaled<T, TKernels>(innerEven, a4, b[10]);
        AddScaled<T, TKernels>(innerEven, a2, b[8]);

        var v = Identity<T>(n, b[0]);
        AddScaled<T, TKernels>(v, a2, b[2]);
        AddScaled<T, TKernels>(v, a4, b[4]);
        AddScaled<T, TKernels>(v, a6, b[6]);

        TKernels.Multiply(ws, a6.ReadOnlyView, innerEven.ReadOnlyView, v.View, T.One, T.One);

        return SolvePade<T, TKernels>(u, v, ws);
    }

    /// <summary>
    /// Solve (V - U) X = (V + U), which is r_m(A).
    ///
    /// The numerator is p_m(A) = U + V and the denominator q_m(A) = p_m(-A) =
    /// V - U, since U holds the odd-degree terms and V the even ones.
    ///
    /// The denominator is a temporary, so it is factored in place, on the
    /// caller's workspace. (The real-only implementation this replaced took a
    /// copy and factored it on the shared workspace, whatever it was given.)
    /// </summary>
    private static Matrix<T> SolvePade<T, TKernels>(Matrix<T> u, Matrix<T> v, Workspace ws)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int n = u.Rows;

        var denominator = new Matrix<T>(n, n);
        var numerator = new Matrix<T>(n, n);

        for (int j = 0; j < n; j++)
        {
            Span<T> uj = u.Column(j);
            Span<T> vj = v.Column(j);
            Span<T> dj = denominator.Column(j);
            Span<T> nj = numerator.Column(j);

            for (int i = 0; i < n; i++)
            {
                dj[i] = vj[i] - uj[i];
                nj[i] = vj[i] + uj[i];
            }
        }

        LuDecomposition<T> lu = TKernels.FactorLuInPlace(ws, denominator);

        if (lu.IsSingular)
        {
            throw new InvalidOperationException(
                $"The Padé denominator is exactly singular at column {lu.SingularColumn}, "
                + "which indicates a non-finite input rather than a property of the algorithm.");
        }

        return lu.Solve(numerator.ReadOnlyView);
    }

    /// <summary>
    /// The <c>ell</c> function of equation (3.4): how many extra halvings the
    /// backward error bound needs beyond what the ||A^k||^(1/k) test suggests.
    ///
    /// It compares ||abs(A)^(2m+1)||_1 against ||A||_1, which detects the case
    /// where cancellation makes the powers of A much smaller than the powers
    /// of its magnitudes -- the situation in which the norm test alone would
    /// choose too low a degree.
    /// </summary>
    private static int Ell<T, TKernels>(Matrix<T> a, int m)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        double aNorm = TKernels.OneNorm(a.ReadOnlyView);
        if (aNorm == 0.0) return 0;

        double absPowerNorm = OneNormOfAbsolutePower<T, TKernels>(a, (2 * m) + 1);
        if (absPowerNorm == 0.0) return 0;

        double alpha = absPowerNorm / (aNorm * AbsoluteCoefficientReciprocal(m));
        int value = (int)Math.Ceiling(Math.Log2(alpha / UnitRoundoff) / (2 * m));

        return Math.Max(value, 0);
    }

    /// <summary>
    /// 1 / |c_{2m+1}|, the reciprocal of the leading coefficient of the
    /// backward error series, which equals C(2m, m) * (2m+1)!.
    ///
    /// Written as literals because the degree-13 value exceeds what an exact
    /// factorial in <c>long</c> could reach, and re-derived by the test suite
    /// from the binomial and the factorial so that a mistyped digit fails
    /// rather than quietly shifting the scaling.
    /// </summary>
    private static double AbsoluteCoefficientReciprocal(int m) => m switch
    {
        3 => 100800.0,
        5 => 10059033600.0,
        7 => 4487938430976000.0,
        9 => 5914384781877411840000.0,
        13 => 113250775606021113483283660800000000.0,
        _ => throw new ArgumentOutOfRangeException(nameof(m), m, "No backward error coefficient for this Padé degree."),
    };

    /// <summary>
    /// ||abs(A)^p||_1, exactly, without forming the power.
    ///
    /// abs(A) is nonnegative, so its powers are too, and the 1-norm of a
    /// nonnegative matrix M is the largest column sum -- that is, the largest
    /// entry of the row vector e^T M. Propagating e^T through p products costs
    /// O(n^2 p), where forming abs(A)^p would cost p products of order n. For
    /// a complex A, abs(A) is the matrix of moduli, as in the paper's complex
    /// case.
    ///
    /// The moduli are taken once, into an n x n real matrix, rather than on
    /// every one of the p passes. For a real matrix that is the same values
    /// summed in the same order, so it changes nothing but the time; for a
    /// complex one it is most of the time, since a robust modulus costs a
    /// division and a square root and p is 27 at degree 13 -- measured, the
    /// per-pass form spent about a fifth of a complex n=256 exponential here.
    /// </summary>
    private static double OneNormOfAbsolutePower<T, TKernels>(Matrix<T> a, int p)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int n = a.Rows;

        double[] magnitudes = Storage.Array<double>(n * n, $"the {n}x{n} moduli of an ell test");
        for (int j = 0; j < n; j++)
        {
            ReadOnlySpan<T> column = a.ReadOnlyView.Column(j);
            Span<double> target = magnitudes.AsSpan(j * n, n);
            for (int i = 0; i < n; i++) target[i] = TKernels.Magnitude(column[i]);
        }

        double[] row = new double[n];
        double[] next = new double[n];
        Array.Fill(row, 1.0);

        for (int step = 0; step < p; step++)
        {
            for (int j = 0; j < n; j++)
            {
                ReadOnlySpan<double> column = magnitudes.AsSpan(j * n, n);

                double sum = 0.0;
                for (int i = 0; i < n; i++) sum += row[i] * column[i];

                next[j] = sum;
            }

            (row, next) = (next, row);
        }

        double best = 0.0;
        for (int j = 0; j < n; j++) best = Math.Max(best, row[j]);

        return best;
    }

    /// <summary>A*B as a new matrix, which is the shape every power here takes.</summary>
    private static Matrix<T> Product<T, TKernels>(Matrix<T> x, Matrix<T> y, Workspace ws)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        var result = new Matrix<T>(x.Rows, y.Columns);
        TKernels.Multiply(ws, x.ReadOnlyView, y.ReadOnlyView, result.View, T.One, T.Zero);
        return result;
    }

    /// <summary>A real multiple of the identity, as a fresh n x n matrix.</summary>
    private static Matrix<T> Identity<T>(int n, double scale)
        where T : unmanaged, INumberBase<T>
    {
        var result = new Matrix<T>(n, n);
        T diagonal = T.CreateTruncating(scale);
        for (int i = 0; i < n; i++) result[i, i] = diagonal;
        return result;
    }

    /// <summary>A fresh copy of <paramref name="source"/> scaled by <paramref name="scale"/>.</summary>
    private static Matrix<T> Scaled<T, TKernels>(Matrix<T> source, double scale)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int rows = source.Rows;
        var result = new Matrix<T>(rows, source.Columns);

        for (int j = 0; j < source.Columns; j++)
        {
            ReadOnlySpan<T> from = source.ReadOnlyView.Column(j);
            Span<T> to = result.Column(j);
            for (int i = 0; i < rows; i++) to[i] = TKernels.Scale(from[i], scale);
        }

        return result;
    }

    /// <summary>target := target + scale * source, both n x n.</summary>
    private static void AddScaled<T, TKernels>(Matrix<T> target, Matrix<T> source, double scale)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        int rows = target.Rows;

        for (int j = 0; j < target.Columns; j++)
        {
            ReadOnlySpan<T> from = source.ReadOnlyView.Column(j);
            Span<T> to = target.Column(j);
            for (int i = 0; i < rows; i++) to[i] += TKernels.Scale(from[i], scale);
        }
    }

    /// <summary>Record the branch taken, when a caller asked for it.</summary>
    private static void Report(ExpmDiagnostics? diagnostics, int degree, int squarings)
    {
        if (diagnostics is null) return;

        diagnostics.Degree = degree;
        diagnostics.Squarings = squarings;
    }
}

/// <summary>
/// Which branch <see cref="MatrixExponential.Expm(Matrix{double}, Workspace)"/>
/// took. Internal: the choice is an implementation detail, observable only so
/// that the suite can prove every Padé degree is reachable and reached.
/// </summary>
internal sealed class ExpmDiagnostics
{
    /// <summary>The Padé degree used, or 0 for the scalar and empty cases.</summary>
    public int Degree { get; set; }

    /// <summary>How many squaring steps undid the scaling; 0 when unscaled.</summary>
    public int Squarings { get; set; }
}
