namespace Tensile.Tests;

/// <summary>
/// The matrix exponential.
///
/// Three layers, because no one of them is sufficient on its own:
///
/// - The coefficient tables are re-derived from their closed forms and
///   compared entry by entry. Those numbers were transcribed from a paper, and
///   a mistyped digit in a Padé coefficient does not crash -- it quietly costs
///   accuracy. Deriving them independently turns that into a failure.
/// - Closed-form exponentials (diagonal, nilpotent, rotation generator) pin
///   the result exactly where the answer is known analytically.
/// - Everything else is checked against a deliberately naive Taylor oracle.
///   The oracle shares nothing with the implementation except GEMM: no Padé,
///   no LU solve, no theta table, no <c>ell</c>. It is far too slow to ship
///   and completely obvious to read, which is exactly what an oracle should
///   be.
///
/// The theta thresholds are the one part with no independent derivation. They
/// are guarded by the oracle comparison, run across norms chosen to reach
/// every Padé degree -- and the branch coverage test proves those norms really
/// do reach all five, so the low-degree paths cannot rot unnoticed.
/// </summary>
public class MatrixExponentialTests
{
    /// <summary>
    /// Relative to the result's own 1-norm. Backward stability does not
    /// promise every digit, and the squaring phase amplifies whatever the
    /// approximant got wrong, so this is loose enough to survive a
    /// well-conditioned exponential rather than tuned to what happens to pass.
    /// </summary>
    private const double Tolerance = 1e-11;

    // ---- coefficient tables ------------------------------------------------

    /// <summary>
    /// The Padé numerator coefficient of x^j is
    /// c_j = (2m-j)! m! / ((2m)! j! (m-j)!); the shipped table is that
    /// normalised so the leading coefficient is 1, which since
    /// c_m = m! / (2m)! reduces to the integer
    ///
    ///     b_j = (2m-j)! / (j! (m-j)!).
    ///
    /// Re-derived here in exact integer arithmetic. The comparison is exact,
    /// not toleranced: every one of these integers carries enough factors of
    /// two to be exactly representable in a double (the largest,
    /// 26!/13! = 64764752532480000, is 7905853580625 * 2^13), which
    /// <see cref="EveryPadeCoefficientIsExactInADouble"/> pins separately.
    /// </summary>
    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(13)]
    public void PadeCoefficientsMatchTheirClosedForm(int m)
    {
        double[] shipped = ShippedPadeCoefficients(m);

        Assert.Equal(m + 1, shipped.Length);

        for (int j = 0; j <= m; j++)
        {
            var derived = Factorial((2 * m) - j) / (Factorial(j) * Factorial(m - j));
            Assert.Equal((double)derived, shipped[j]);
        }
    }

    /// <summary>
    /// The table is written as integers on the claim that they are exact in a
    /// double. Stated as its own test because the claim is not obvious -- all
    /// but the smallest exceed 2^53 -- and it is what licenses the exact
    /// comparison above.
    /// </summary>
    [Theory]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(7)]
    [InlineData(9)]
    [InlineData(13)]
    public void EveryPadeCoefficientIsExactInADouble(int m)
    {
        foreach (double value in ShippedPadeCoefficients(m))
        {
            var exact = new System.Numerics.BigInteger(value);
            Assert.Equal(value, (double)exact);
            Assert.Equal(0.0, value - (double)exact);
        }
    }

    /// <summary>
    /// The <c>ell</c> coefficients are 1/|c_{2m+1}| = C(2m, m) * (2m+1)!.
    ///
    /// Unlike the Padé table these are NOT all exactly representable -- m = 9
    /// and m = 13 are not -- so the comparison is to within a rounding step
    /// rather than exact. That still catches a mistyped digit, because any
    /// digit that survives into the double changes it by far more than an ulp,
    /// and a digit that does not survive cannot affect the computation either.
    /// </summary>
    [Theory]
    [InlineData(3, 100800.0)]
    [InlineData(5, 10059033600.0)]
    [InlineData(7, 4487938430976000.0)]
    [InlineData(9, 5914384781877411840000.0)]
    [InlineData(13, 113250775606021113483283660800000000.0)]
    public void BackwardErrorCoefficientsMatchTheirClosedForm(int m, double shipped)
    {
        var binomial = Factorial(2 * m) / (Factorial(m) * Factorial(m));
        var derived = (double)(binomial * Factorial((2 * m) + 1));

        Assert.True(
            Math.Abs(derived - shipped) <= 1e-15 * Math.Abs(derived),
            $"m={m}: derived {derived:E17} against shipped {shipped:E17}");
    }

    // ---- closed-form exponentials ------------------------------------------

    [Fact]
    public void ExponentialOfZeroIsTheIdentity()
    {
        var result = Matrix.Zeros<double>(6, 6).Expm();

        for (int j = 0; j < 6; j++)
            for (int i = 0; i < 6; i++)
                Assert.Equal(i == j ? 1.0 : 0.0, result[i, j]);
    }

    /// <summary>exp of a diagonal matrix is the diagonal of the scalar exponentials.</summary>
    [Theory]
    [InlineData(1e-4)]
    [InlineData(0.5)]
    [InlineData(3.0)]
    [InlineData(20.0)]
    public void ExponentialOfADiagonalIsElementWise(double scale)
    {
        double[] diagonal = [0.0, 1.0, -1.0, 0.25, -2.0];
        int n = diagonal.Length;

        var a = Matrix.Zeros<double>(n, n);
        for (int i = 0; i < n; i++) a[i, i] = diagonal[i] * scale;

        var result = a.Expm();

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                double expected = i == j ? Math.Exp(diagonal[i] * scale) : 0.0;
                Assert.True(
                    Math.Abs(result[i, j] - expected) <= Tolerance * Math.Max(Math.Abs(expected), 1.0),
                    $"[{i},{j}] {result[i, j]:E17} vs {expected:E17}");
            }
        }
    }

    /// <summary>
    /// A nilpotent matrix has a terminating series: for the 2x2 single Jordan
    /// block with zero eigenvalue, exp(A) = I + A exactly.
    /// </summary>
    [Fact]
    public void ExponentialOfANilpotentBlockTerminates()
    {
        var a = Matrix.FromRows(new double[,] { { 0, 1 }, { 0, 0 } });

        var result = a.Expm();

        Assert.Equal(1.0, result[0, 0], 15);
        Assert.Equal(1.0, result[0, 1], 15);
        Assert.Equal(0.0, result[1, 0], 15);
        Assert.Equal(1.0, result[1, 1], 15);
    }

    /// <summary>
    /// The 2x2 rotation generator: exp([[0,-t],[t,0]]) is the rotation by t.
    /// Chosen because it is a genuinely nonsymmetric case with a closed form,
    /// and because its exponential is orthogonal, so any error shows up as a
    /// loss of orthogonality rather than hiding in a norm.
    /// </summary>
    [Theory]
    [InlineData(0.001)]
    [InlineData(0.3)]
    [InlineData(1.0)]
    [InlineData(7.5)]
    public void ExponentialOfTheRotationGeneratorIsARotation(double t)
    {
        var a = Matrix.FromRows(new double[,] { { 0, -t }, { t, 0 } });

        var result = a.Expm();

        Assert.Equal(Math.Cos(t), result[0, 0], 12);
        Assert.Equal(-Math.Sin(t), result[0, 1], 12);
        Assert.Equal(Math.Sin(t), result[1, 0], 12);
        Assert.Equal(Math.Cos(t), result[1, 1], 12);
    }

    /// <summary>
    /// A Jordan block with a nonzero eigenvalue:
    /// exp([[a,1],[0,a]]) = e^a [[1,1],[0,1]].
    /// </summary>
    [Fact]
    public void ExponentialOfAJordanBlockIsExact()
    {
        const double Lambda = 1.75;
        var a = Matrix.FromRows(new double[,] { { Lambda, 1 }, { 0, Lambda } });

        var result = a.Expm();
        double e = Math.Exp(Lambda);

        Assert.Equal(e, result[0, 0], 11);
        Assert.Equal(e, result[0, 1], 11);
        Assert.Equal(0.0, result[1, 0], 11);
        Assert.Equal(e, result[1, 1], 11);
    }

    // ---- against the oracle ------------------------------------------------

    /// <summary>
    /// The main accuracy check: random matrices scaled across the whole range
    /// of norms the degree ladder covers, compared against the Taylor oracle.
    /// </summary>
    [Theory]
    [InlineData(8, 0.001)]
    [InlineData(8, 0.05)]
    [InlineData(8, 0.4)]
    [InlineData(8, 1.2)]
    [InlineData(8, 6.0)]
    [InlineData(8, 40.0)]
    [InlineData(17, 0.01)]
    [InlineData(17, 0.8)]
    [InlineData(17, 12.0)]
    [InlineData(32, 2.5)]
    [InlineData(32, 30.0)]
    public void AgreesWithTheTaylorOracle(int n, double targetNorm)
    {
        var a = RandomWithOneNorm(n, targetNorm, seed: (n * 1000) + (int)(targetNorm * 10));

        var actual = a.Expm();
        var expected = TaylorOracle(a);

        double relative = RelativeDifference(actual, expected);

        Assert.True(
            relative <= Tolerance,
            $"n={n}, ||A||_1~{targetNorm}: relative difference {relative:E3} exceeds {Tolerance:E3}");
    }

    /// <summary>
    /// A nonnormal matrix, which is the case the 2009 algorithm exists for.
    /// Its powers are far smaller than its norm suggests, so choosing the
    /// scaling from ||A|| alone would overscale badly.
    /// </summary>
    [Theory]
    [InlineData(10, 1.0)]
    [InlineData(10, 25.0)]
    [InlineData(24, 8.0)]
    public void AgreesWithTheOracleOnAStrictlyTriangularMatrix(int n, double scale)
    {
        // Strictly upper triangular is nilpotent, so every eigenvalue is zero
        // while the norm is whatever we make it -- maximum disagreement
        // between ||A|| and the ||A^k||^(1/k) the algorithm actually uses.
        var a = Matrix.Zeros<double>(n, n);
        var rng = new Random(4242);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < j; i++)
                a[i, j] = ((rng.NextDouble() * 2.0) - 1.0) * scale;

        var actual = a.Expm();
        var expected = TaylorOracle(a);

        double relative = RelativeDifference(actual, expected);

        Assert.True(relative <= Tolerance, $"n={n}, scale={scale}: relative difference {relative:E3}");
    }

    // ---- branch coverage ---------------------------------------------------

    /// <summary>
    /// Every Padé degree must be reachable, and the accuracy suite above must
    /// actually reach them. Without this, a suite whose matrices all landed on
    /// degree 13 would leave four branches unexecuted and still pass --
    /// finding 10, applied to branches rather than to files.
    /// </summary>
    [Fact]
    public void EveryPadeDegreeIsReached()
    {
        double[] norms = [0.001, 0.05, 0.4, 1.2, 6.0, 40.0];
        var seen = new HashSet<int>();

        foreach (double norm in norms)
        {
            var a = RandomWithOneNorm(8, norm, seed: 77 + (int)(norm * 10));

            var diagnostics = new ExpmDiagnostics();
            _ = MatrixExponential.Expm(a, workspace: null, diagnostics);

            seen.Add(diagnostics.Degree);
        }

        Assert.Equal([3, 5, 7, 9, 13], seen.Order().ToArray());
    }

    /// <summary>
    /// A large norm must scale, and a small one must not. This pins the
    /// direction of the scaling decision, which a sign error in the log2 would
    /// invert while leaving small cases correct.
    /// </summary>
    [Fact]
    public void ScalingHappensOnlyWhenTheNormDemandsIt()
    {
        var small = RandomWithOneNorm(6, 0.05, seed: 11);
        var large = RandomWithOneNorm(6, 500.0, seed: 12);

        var smallDiagnostics = new ExpmDiagnostics();
        _ = MatrixExponential.Expm(small, workspace: null, smallDiagnostics);

        var largeDiagnostics = new ExpmDiagnostics();
        _ = MatrixExponential.Expm(large, workspace: null, largeDiagnostics);

        Assert.Equal(0, smallDiagnostics.Squarings);
        Assert.True(largeDiagnostics.Squarings >= 6, $"expected several squarings, got {largeDiagnostics.Squarings}");
    }

    // ---- identities --------------------------------------------------------

    /// <summary>
    /// A and -A commute, so exp(A) exp(-A) = I for every A. This is a property
    /// the algorithm is never told about, and it exercises the result rather
    /// than the path.
    /// </summary>
    [Theory]
    [InlineData(0.2)]
    [InlineData(3.0)]
    [InlineData(15.0)]
    public void ExponentialTimesItsInverseIsTheIdentity(double norm)
    {
        const int N = 12;

        var a = RandomWithOneNorm(N, norm, seed: 909);
        var negated = Scaled(a, -1.0);

        var product = a.Expm().Multiply(negated.Expm().ReadOnlyView);

        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < N; i++)
            {
                double expected = i == j ? 1.0 : 0.0;
                Assert.True(
                    Math.Abs(product[i, j] - expected) <= 1e-9,
                    $"[{i},{j}] = {product[i, j]:E17}, expected {expected}");
            }
        }
    }

    /// <summary>exp(A^T) = exp(A)^T, since every term of the series transposes.</summary>
    [Fact]
    public void ExponentialCommutesWithTransposition()
    {
        const int N = 9;

        var a = RandomWithOneNorm(N, 4.0, seed: 313);
        var transposed = Transpose(a);

        var left = transposed.Expm();
        var right = Transpose(a.Expm());

        Assert.True(RelativeDifference(left, right) <= Tolerance);
    }

    /// <summary>
    /// exp((s+t)A) = exp(sA) exp(tA), because sA and tA commute. A stronger
    /// check than it looks: the two sides usually take different branches of
    /// the degree ladder and different scaling counts.
    /// </summary>
    [Fact]
    public void ExponentialIsAOneParameterGroup()
    {
        const int N = 10;

        var a = RandomWithOneNorm(N, 3.0, seed: 515);

        var whole = Scaled(a, 1.25).Expm();
        var split = Scaled(a, 0.5).Expm().Multiply(Scaled(a, 0.75).Expm().ReadOnlyView);

        Assert.True(RelativeDifference(whole, split) <= 1e-10);
    }

    // ---- degenerate shapes and arguments -----------------------------------

    [Fact]
    public void EmptyMatrixExponentiatesToItself()
    {
        var result = Matrix.Zeros<double>(0, 0).Expm();

        Assert.Equal(0, result.Rows);
        Assert.Equal(0, result.Columns);
    }

    [Fact]
    public void OneByOneIsScalarExponentiation()
    {
        var a = Matrix.Zeros<double>(1, 1);
        a[0, 0] = -3.25;

        Assert.Equal(Math.Exp(-3.25), a.Expm()[0, 0], 15);
    }

    [Fact]
    public void RectangularIsRejected()
    {
        var a = Matrix.Zeros<double>(3, 4);

        var error = Assert.Throws<ArgumentException>(() => a.Expm());
        Assert.Contains("square", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void NullIsRejected()
    {
        Matrix<double>? a = null;
        Assert.Throws<ArgumentNullException>(() => a!.Expm());
    }

    /// <summary>The input must survive the call; every power is formed into fresh storage.</summary>
    [Fact]
    public void TheInputIsNotModified()
    {
        var a = RandomWithOneNorm(7, 5.0, seed: 2024);
        var before = a.Clone();

        _ = a.Expm();

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                Assert.Equal(before[i, j], a[i, j]);
    }

    // ---- the oracle and other helpers --------------------------------------

    /// <summary>
    /// exp(A) by scaling and squaring with a truncated Taylor series.
    ///
    /// Deliberately the dumbest correct algorithm available. The scaled matrix
    /// has 1-norm at most 1/32, so the first omitted term is bounded by
    /// (1/32)^41 / 41!, which is smaller than anything double precision can
    /// represent relative to the leading identity -- truncation is not a
    /// source of error here, only rounding is.
    ///
    /// It shares GEMM with the implementation and nothing else: no Padé
    /// approximant, no linear solve, no theta table, no <c>ell</c> correction.
    /// </summary>
    private static Matrix<double> TaylorOracle(Matrix<double> a, int terms = 40)
    {
        int n = a.Rows;

        int s = 0;
        double norm = a.OneNorm();
        while (norm > 0.03125)
        {
            norm *= 0.5;
            s++;
        }

        var b = Scaled(a, Math.ScaleB(1.0, -s));

        var result = Matrix.Identity<double>(n);
        var term = Matrix.Identity<double>(n);

        for (int k = 1; k <= terms; k++)
        {
            term = Scaled(term.Multiply(b.ReadOnlyView), 1.0 / k);
            AddInto(result, term);
        }

        for (int i = 0; i < s; i++) result = result.Multiply(result.ReadOnlyView);

        return result;
    }

    /// <summary>A random matrix scaled so that its exact 1-norm is the target.</summary>
    private static Matrix<double> RandomWithOneNorm(int n, double target, int seed)
    {
        var rng = new Random(seed);
        var a = Matrix.Zeros<double>(n, n);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                a[i, j] = (rng.NextDouble() * 2.0) - 1.0;

        double norm = a.OneNorm();
        return Scaled(a, target / norm);
    }

    private static Matrix<double> Scaled(Matrix<double> a, double scale)
    {
        var result = Matrix.Zeros<double>(a.Rows, a.Columns);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[i, j] = a[i, j] * scale;

        return result;
    }

    private static void AddInto(Matrix<double> target, Matrix<double> source)
    {
        for (int j = 0; j < target.Columns; j++)
            for (int i = 0; i < target.Rows; i++)
                target[i, j] += source[i, j];
    }

    private static Matrix<double> Transpose(Matrix<double> a)
    {
        var result = Matrix.Zeros<double>(a.Columns, a.Rows);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[j, i] = a[i, j];

        return result;
    }

    /// <summary>||X - Y||_1 / ||Y||_1, floored so an all-zero reference cannot divide by zero.</summary>
    private static double RelativeDifference(Matrix<double> x, Matrix<double> y)
    {
        double difference = 0.0;
        double reference = 0.0;

        for (int j = 0; j < x.Columns; j++)
        {
            double columnDifference = 0.0;
            double columnReference = 0.0;

            for (int i = 0; i < x.Rows; i++)
            {
                columnDifference += Math.Abs(x[i, j] - y[i, j]);
                columnReference += Math.Abs(y[i, j]);
            }

            difference = Math.Max(difference, columnDifference);
            reference = Math.Max(reference, columnReference);
        }

        return difference / Math.Max(reference, 1.0);
    }

    /// <summary>The shipped table, reached by reflection so the fields stay private.</summary>
    private static double[] ShippedPadeCoefficients(int m)
    {
        var field = typeof(MatrixExponential).GetField(
            $"Pade{m}",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);

        Assert.NotNull(field);
        return (double[])field!.GetValue(null)!;
    }

    private static System.Numerics.BigInteger Factorial(int k)
    {
        var result = System.Numerics.BigInteger.One;
        for (int i = 2; i <= k; i++) result *= i;
        return result;
    }
}
