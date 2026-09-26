using System.Numerics;

namespace Tensile.Tests;

/// <summary>
/// Complex exponentials, computed through the real embedding.
///
/// The oracle here is a Taylor series written directly in complex arithmetic,
/// with its own naive complex matrix product. It shares nothing with the code
/// under test: no embedding, no real <c>Expm</c>, no GEMM. Agreement therefore
/// checks the embedding's sign convention and the projection back, not merely
/// that the real algorithm ran.
///
/// The invariant tests matter as much as the oracle, because they hold without
/// knowing the answer: exp(iH) is unitary for Hermitian H, exp(A)exp(-A) is the
/// identity, and exp(tA)b preserves the 2-norm of b when A is skew-Hermitian.
/// That last one is the oscillatory, lossless case, whose trace is purely
/// imaginary -- the part of the shift the real representation cannot remove,
/// and which the complex overload therefore removes itself.
/// </summary>
public class ComplexExponentialTests
{
    private const double Tolerance = 1e-11;

    // ---- the embedding itself -----------------------------------------------

    /// <summary>
    /// Embed(A) Embed(B) = Embed(AB). This is the property the whole approach
    /// rests on, and it pins the sign convention: the transposed convention
    /// [[X, Y], [-Y, X]] represents conj(A) and would pass a test of the map
    /// being linear, but not this one against a complex product.
    /// </summary>
    [Fact]
    public void TheEmbeddingIsMultiplicative()
    {
        var a = RandomComplex(7, 7, seed: 11);
        var b = RandomComplex(7, 7, seed: 12);

        var left = ComplexEmbedding.Embed(a).Multiply(ComplexEmbedding.Embed(b).ReadOnlyView);
        var right = ComplexEmbedding.Embed(Multiply(a, b));

        for (int j = 0; j < 14; j++)
            for (int i = 0; i < 14; i++)
                Assert.Equal(right[i, j], left[i, j], 12);
    }

    /// <summary>Embed(A) Stack(b) = Stack(Ab), which is what the action relies on.</summary>
    [Fact]
    public void TheEmbeddingActsOnTheStack()
    {
        var a = RandomComplex(6, 6, seed: 13);
        var b = RandomComplex(6, 3, seed: 14);

        var left = ComplexEmbedding.Embed(a).Multiply(ComplexEmbedding.Stack(b.ReadOnlyView).ReadOnlyView);
        var right = ComplexEmbedding.Stack(Multiply(a, b).ReadOnlyView);

        for (int j = 0; j < 3; j++)
            for (int i = 0; i < 12; i++)
                Assert.Equal(right[i, j], left[i, j], 12);
    }

    /// <summary>
    /// Projecting an exact embedding recovers the matrix exactly -- averaging
    /// two identical copies is exact in floating point -- and unstacking a
    /// stack is the identity.
    /// </summary>
    [Fact]
    public void TheRoundTripsAreExact()
    {
        var a = RandomComplex(5, 5, seed: 15);
        var b = RandomComplex(5, 2, seed: 16);

        var projected = ComplexEmbedding.Project(ComplexEmbedding.Embed(a));
        var unstacked = ComplexEmbedding.Unstack(ComplexEmbedding.Stack(b.ReadOnlyView));

        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 5; i++)
                Assert.Equal(a[i, j], projected[i, j]);

        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 5; i++)
                Assert.Equal(b[i, j], unstacked[i, j]);
    }

    /// <summary>
    /// The real algorithm knows nothing of the block structure, so rounding
    /// perturbs the paired blocks independently. They must still agree to
    /// near working precision, or the projection would be averaging two
    /// different answers rather than two roundings of one.
    /// </summary>
    [Theory]
    [InlineData(0.5)]
    [InlineData(6.0)]
    [InlineData(40.0)]
    public void TheComputedExponentialStaysOnTheComplexStructure(double norm)
    {
        var a = WithOneNorm(RandomComplex(10, 10, seed: 17), norm);

        var real = ComplexEmbedding.Embed(a).Expm();
        double defect = ComplexEmbedding.StructuralDefect(real);

        Assert.True(defect <= 1e-12, $"||A||={norm}: structural defect {defect:E3}");
    }

    // ---- Expm against the complex Taylor oracle -----------------------------

    [Theory]
    [InlineData(6, 0.01)]
    [InlineData(6, 0.5)]
    [InlineData(6, 3.0)]
    [InlineData(6, 25.0)]
    [InlineData(12, 1.0)]
    [InlineData(12, 10.0)]
    [InlineData(20, 4.0)]
    [InlineData(20, 40.0)]
    public void ExpmAgreesWithAnIndependentComplexTaylorSeries(int n, double norm)
    {
        var a = WithOneNorm(RandomComplex(n, n, seed: (n * 101) + (int)(norm * 7)), norm);

        var actual = a.Expm();
        var expected = TaylorOracle(a);

        double relative = RelativeDifference(actual, expected);
        Assert.True(relative <= Tolerance, $"n={n}, ||A||={norm}: {relative:E3}");
    }

    /// <summary>
    /// A real matrix embeds block-diagonally, with the same norms as itself,
    /// so the complex path must choose exactly the real path's degree and
    /// scaling and reach the real answer with a zero imaginary part. This is
    /// the check that the embedding costs nothing when there is nothing
    /// complex to represent.
    /// </summary>
    [Theory]
    [InlineData(0.05)]
    [InlineData(1.2)]
    [InlineData(40.0)]
    public void ARealMatrixTakesTheRealPath(double norm)
    {
        const int N = 9;

        var real = WithOneNorm(RandomReal(N, seed: 18), norm);
        var complex = ToComplex(real);

        var realDiagnostics = new ExpmDiagnostics();
        var expected = MatrixExponential.Expm(real, workspace: null, realDiagnostics);

        var complexDiagnostics = new ExpmDiagnostics();
        var actual = MatrixExponential.Expm(complex, workspace: null, complexDiagnostics);

        Assert.Equal(realDiagnostics.Degree, complexDiagnostics.Degree);
        Assert.Equal(realDiagnostics.Squarings, complexDiagnostics.Squarings);

        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < N; i++)
            {
                Assert.Equal(expected[i, j], actual[i, j].Real, 12);
                Assert.Equal(0.0, actual[i, j].Imaginary, 12);
            }
        }
    }

    // ---- Expm closed forms and invariants -----------------------------------

    /// <summary>exp of a complex diagonal is the diagonal of scalar exponentials, oscillation included.</summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(8.0)]
    public void DiagonalIsElementWise(double scale)
    {
        Complex[] diagonal =
        [
            Complex.Zero, new(1, 2), new(-1, -3), new(0, 0.5), new(-2, 0.25),
        ];

        int n = diagonal.Length;
        var a = Matrix.Zeros<Complex>(n, n);
        for (int i = 0; i < n; i++) a[i, i] = diagonal[i] * scale;

        var result = a.Expm();

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < n; i++)
            {
                Complex expected = i == j ? Complex.Exp(diagonal[i] * scale) : Complex.Zero;
                Assert.True(
                    Complex.Abs(result[i, j] - expected) <= Tolerance * Math.Max(Complex.Abs(expected), 1.0),
                    $"[{i},{j}] {result[i, j]} vs {expected}");
            }
        }
    }

    /// <summary>exp([[lambda, 1], [0, lambda]]) = e^lambda [[1, 1], [0, 1]] for complex lambda.</summary>
    [Fact]
    public void JordanBlockWithAComplexEigenvalue()
    {
        var lambda = new Complex(0.75, -2.5);

        var a = Matrix.Zeros<Complex>(2, 2);
        a[0, 0] = lambda;
        a[0, 1] = Complex.One;
        a[1, 1] = lambda;

        var result = a.Expm();
        Complex e = Complex.Exp(lambda);

        Assert.True(Complex.Abs(result[0, 0] - e) <= Tolerance);
        Assert.True(Complex.Abs(result[0, 1] - e) <= Tolerance);
        Assert.True(Complex.Abs(result[1, 0]) <= Tolerance);
        Assert.True(Complex.Abs(result[1, 1] - e) <= Tolerance);
    }

    /// <summary>
    /// exp(iH) is unitary for Hermitian H. An invariant that needs no oracle,
    /// and a natural one for this application: a lossless line's propagation
    /// is exactly this shape.
    /// </summary>
    [Theory]
    [InlineData(0.5)]
    [InlineData(5.0)]
    [InlineData(30.0)]
    public void ExponentialOfIHermitianIsUnitary(double norm)
    {
        const int N = 10;

        var h = Hermitian(N, seed: 19);
        var a = WithOneNorm(Scaled(h, Complex.ImaginaryOne), norm);

        var u = a.Expm();
        var gram = Multiply(ConjugateTranspose(u), u);

        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < N; i++)
            {
                Complex expected = i == j ? Complex.One : Complex.Zero;
                Assert.True(
                    Complex.Abs(gram[i, j] - expected) <= Tolerance,
                    $"||A||={norm}: (U^H U)[{i},{j}] = {gram[i, j]}");
            }
        }
    }

    /// <summary>A and -A commute, so exp(A) exp(-A) is the identity.</summary>
    [Theory]
    [InlineData(0.3)]
    [InlineData(4.0)]
    [InlineData(20.0)]
    public void ExponentialTimesItsNegationIsTheIdentity(double norm)
    {
        const int N = 8;

        var a = WithOneNorm(RandomComplex(N, N, seed: 20), norm);
        var product = Multiply(a.Expm(), Scaled(a, -Complex.One).Expm());

        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < N; i++)
            {
                Complex expected = i == j ? Complex.One : Complex.Zero;
                Assert.True(Complex.Abs(product[i, j] - expected) <= 1e-9, $"[{i},{j}] = {product[i, j]}");
            }
        }
    }

    // ---- Expmv ---------------------------------------------------------------

    [Theory]
    [InlineData(8, 0.1, 1)]
    [InlineData(8, 5.0, 3)]
    [InlineData(16, 1.0, 1)]
    [InlineData(16, 20.0, 2)]
    [InlineData(24, 8.0, 1)]
    public void ExpmvAgreesWithExpmTimesB(int n, double norm, int columns)
    {
        var a = WithOneNorm(RandomComplex(n, n, seed: (n * 37) + (int)norm), norm);
        var b = RandomComplex(n, columns, seed: 21);

        var actual = a.Expmv(b.ReadOnlyView);
        var expected = Multiply(TaylorOracle(a), b);

        double relative = RelativeDifference(actual, expected);
        Assert.True(relative <= Tolerance, $"n={n}, ||A||={norm}: {relative:E3}");
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1e-3)]
    [InlineData(0.5)]
    [InlineData(2.5)]
    [InlineData(-1.0)]
    [InlineData(-6.0)]
    public void ExpmvHandlesTheTimeArgument(double t)
    {
        const int N = 10;

        var a = WithOneNorm(RandomComplex(N, N, seed: 22), 3.0);
        var b = RandomComplex(N, 2, seed: 23);

        var actual = a.Expmv(b.ReadOnlyView, t);
        var expected = Multiply(TaylorOracle(Scaled(a, t)), b);

        double relative = RelativeDifference(actual, expected);
        Assert.True(relative <= Tolerance, $"t={t}: {relative:E3}");
    }

    /// <summary>
    /// exp(tA) is unitary when A is skew-Hermitian, so it preserves the 2-norm
    /// of every column of B. This is the oscillatory, lossless case, where the
    /// trace is purely imaginary and so goes entirely through the imaginary
    /// shift the complex overload applies before embedding.
    /// </summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(25.0)]
    [InlineData(200.0)]
    public void ExpmvPreservesNormForASkewHermitianOperator(double norm)
    {
        const int N = 16;

        var a = WithOneNorm(Scaled(Hermitian(N, seed: 24), Complex.ImaginaryOne), norm);
        var b = RandomComplex(N, 2, seed: 25);

        var y = a.Expmv(b.ReadOnlyView);

        for (int j = 0; j < 2; j++)
        {
            double before = ColumnTwoNorm(b, j);
            double after = ColumnTwoNorm(y, j);

            Assert.True(
                Math.Abs(after - before) <= Tolerance * before,
                $"||A||={norm}, column {j}: {before:E17} -> {after:E17}");
        }
    }

    /// <summary>
    /// Adding i*c*I to the operator must cost nothing: the complex overload
    /// removes the imaginary part of the trace shift before embedding, so the
    /// real algorithm sees the same matrix either way and must choose the same
    /// parameters, and the answer differs by exactly the rotation e^(ict).
    ///
    /// This is a regression test with a measured history. Before the shift was
    /// split, the real representation kept the whole imaginary diagonal, and
    /// this exact case -- norm 5, c = 100 -- took 561 applications against 23.
    /// </summary>
    [Theory]
    [InlineData(20.0, 1.0)]
    [InlineData(100.0, 1.0)]
    [InlineData(100.0, -0.75)]
    public void AnImaginaryShiftCostsNothing(double c, double t)
    {
        const int N = 24;

        var a = WithOneNorm(Scaled(Hermitian(N, seed: 30), Complex.ImaginaryOne), 5.0);
        var withShift = a.Clone();
        for (int i = 0; i < N; i++) withShift[i, i] += new Complex(0.0, c);

        var b = RandomComplex(N, 1, seed: 31);

        var plainDiagnostics = new ExpmvDiagnostics();
        var plain = MatrixExponentialAction.Expmv(a, b.ReadOnlyView, t, plainDiagnostics);

        var shiftedDiagnostics = new ExpmvDiagnostics();
        var shifted = MatrixExponentialAction.Expmv(withShift, b.ReadOnlyView, t, shiftedDiagnostics);

        Assert.Equal(plainDiagnostics.Degree, shiftedDiagnostics.Degree);
        Assert.Equal(plainDiagnostics.Scaling, shiftedDiagnostics.Scaling);

        Complex rotation = Complex.FromPolarCoordinates(1.0, c * t);

        for (int i = 0; i < N; i++)
        {
            Complex expected = plain[i, 0] * rotation;
            Assert.True(
                Complex.Abs(shifted[i, 0] - expected) <= Tolerance * Math.Max(Complex.Abs(expected), 1.0),
                $"c={c}, t={t}, [{i}]: {shifted[i, 0]} vs {expected}");
        }
    }

    /// <summary>
    /// A real operator and real B must reproduce the real Expmv, parameters
    /// included: the block-diagonal embedding has the same norms and the same
    /// trace per row as the real matrix.
    /// </summary>
    [Fact]
    public void ExpmvOnARealMatrixTakesTheRealPath()
    {
        const int N = 12;

        var real = WithOneNorm(RandomReal(N, seed: 26), 6.0);
        var realB = RandomReal(N, seed: 27);

        var realDiagnostics = new ExpmvDiagnostics();
        var expected = MatrixExponentialAction.Expmv(real, realB.ReadOnlyView, 1.0, realDiagnostics);

        var complexDiagnostics = new ExpmvDiagnostics();
        var actual = MatrixExponentialAction.Expmv(
            ToComplex(real), ToComplex(realB).ReadOnlyView, 1.0, complexDiagnostics);

        Assert.Equal(realDiagnostics.Degree, complexDiagnostics.Degree);
        Assert.Equal(realDiagnostics.Scaling, complexDiagnostics.Scaling);

        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < N; i++)
            {
                Assert.Equal(expected[i, j], actual[i, j].Real, 11);
                Assert.Equal(0.0, actual[i, j].Imaginary, 11);
            }
        }
    }

    // ---- the matrix-free complex path -------------------------------------------

    /// <summary>
    /// The operator overload must reach the dense overload's answer by a
    /// different route: no trace shift, no sharp estimates, a supplied bound,
    /// and every application made through the complex operator itself rather
    /// than through a formed real matrix.
    /// </summary>
    [Theory]
    [InlineData(8, 0.5)]
    [InlineData(8, 6.0)]
    [InlineData(20, 15.0)]
    public void MatrixFreeAgreesWithTheDensePath(int n, double norm)
    {
        var a = WithOneNorm(RandomComplex(n, n, seed: 40 + n), norm);
        var b = RandomComplex(n, 2, seed: 41);

        var dense = a.Expmv(b.ReadOnlyView);
        var free = MatrixExponentialAction.Expmv(
            new ApplyOnlyComplexOperator(a), b.ReadOnlyView, t: 1.0, oneNormBound: ComplexOneNorm(a));

        double relative = RelativeDifference(free, dense);
        Assert.True(relative <= Tolerance, $"n={n}, ||A||={norm}: {relative:E3}");
    }

    [Theory]
    [InlineData(-2.0)]
    [InlineData(0.0)]
    [InlineData(3.5)]
    public void MatrixFreeHandlesTheTimeArgument(double t)
    {
        const int N = 10;

        var a = WithOneNorm(RandomComplex(N, N, seed: 42), 2.0);
        var b = RandomComplex(N, 1, seed: 43);

        var free = MatrixExponentialAction.Expmv(
            new ApplyOnlyComplexOperator(a), b.ReadOnlyView, t, ComplexOneNorm(a));
        var expected = Multiply(TaylorOracle(Scaled(a, t)), b);

        Assert.True(RelativeDifference(free, expected) <= Tolerance);
    }

    /// <summary>The unitary invariant, reached through an operator that exposes nothing but Apply.</summary>
    [Fact]
    public void MatrixFreePreservesNormForASkewHermitianOperator()
    {
        const int N = 16;

        var a = WithOneNorm(Scaled(Hermitian(N, seed: 44), Complex.ImaginaryOne), 30.0);
        var b = RandomComplex(N, 1, seed: 45);

        var y = MatrixExponentialAction.Expmv(
            new ApplyOnlyComplexOperator(a), b.ReadOnlyView, 1.0, ComplexOneNorm(a));

        double before = ColumnTwoNorm(b, 0);
        Assert.True(Math.Abs(ColumnTwoNorm(y, 0) - before) <= Tolerance * before);
    }

    /// <summary>
    /// An operator reports whatever order it likes, and twice that order sizes
    /// the real representation. An order whose double does not fit an int must
    /// be an argument error, checked before anything is sized from it -- not an
    /// OverflowException from the checked arithmetic, which is the wrong
    /// diagnosis for a bad argument.
    /// </summary>
    [Fact]
    public void MatrixFreeRejectsAnOrderWhoseRealRepresentationCannotExist()
    {
        var b = RandomComplex(4, 1, seed: 46);

        Assert.Throws<ArgumentOutOfRangeException>(() => MatrixExponentialAction.Expmv(
            new HostileOrderOperator(int.MaxValue / 2 + 1), b.ReadOnlyView, 1.0, 1.0));
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void MatrixFreeRejectsAnUnusableNormBound(double bound)
    {
        var a = Matrix.Identity<Complex>(4);
        var b = RandomComplex(4, 1, seed: 47);

        Assert.Throws<ArgumentOutOfRangeException>(() => MatrixExponentialAction.Expmv(
            new ApplyOnlyComplexOperator(a), b.ReadOnlyView, 1.0, bound));
    }

    [Fact]
    public void MatrixFreeRejectsMismatchedRowsAndNull()
    {
        var a = Matrix.Identity<Complex>(4);
        var b = RandomComplex(5, 1, seed: 48);

        Assert.Throws<ArgumentException>(() => MatrixExponentialAction.Expmv(
            new ApplyOnlyComplexOperator(a), b.ReadOnlyView, 1.0, 1.0));
        Assert.Throws<ArgumentNullException>(() => MatrixExponentialAction.Expmv(
            (ILinearOperator<Complex>)null!, b.ReadOnlyView, 1.0, 1.0));
    }

    // ---- degenerate shapes and arguments ------------------------------------

    [Fact]
    public void EmptyExpmIsEmpty()
    {
        var result = Matrix.Zeros<Complex>(0, 0).Expm();

        Assert.Equal(0, result.Rows);
        Assert.Equal(0, result.Columns);
    }

    [Fact]
    public void OneByOneIsScalarExponentiation()
    {
        var a = Matrix.Zeros<Complex>(1, 1);
        a[0, 0] = new Complex(-0.5, 3.0);

        Assert.Equal(Complex.Exp(new Complex(-0.5, 3.0)), a.Expm()[0, 0]);
    }

    [Fact]
    public void RectangularIsRejected()
    {
        var a = Matrix.Zeros<Complex>(3, 4);
        var b = Matrix.Zeros<Complex>(3, 1);

        Assert.Contains("square", Assert.Throws<ArgumentException>(() => a.Expm()).Message, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("square", Assert.Throws<ArgumentException>(() => a.Expmv(b.ReadOnlyView)).Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void NullIsRejected()
    {
        Matrix<Complex>? a = null;
        var b = Matrix.Zeros<Complex>(3, 1);

        Assert.Throws<ArgumentNullException>(() => a!.Expm());
        Assert.Throws<ArgumentNullException>(() => a!.Expmv(b.ReadOnlyView));
    }

    [Fact]
    public void ExpmvRejectsMismatchedRows()
    {
        var a = Matrix.Identity<Complex>(4);
        var b = Matrix.Zeros<Complex>(5, 1);

        Assert.Throws<ArgumentException>(() => a.Expmv(b.ReadOnlyView));
    }

    [Fact]
    public void ExpmvWithNoColumnsReturnsThemUnchanged()
    {
        var a = Matrix.Identity<Complex>(4);
        var b = Matrix.Zeros<Complex>(4, 0);

        var result = a.Expmv(b.ReadOnlyView);

        Assert.Equal(4, result.Rows);
        Assert.Equal(0, result.Columns);
    }

    [Fact]
    public void TheInputsAreNotModified()
    {
        var a = WithOneNorm(RandomComplex(6, 6, seed: 28), 4.0);
        var b = RandomComplex(6, 2, seed: 29);

        var aBefore = a.Clone();
        var bBefore = b.Clone();

        _ = a.Expm();
        _ = a.Expmv(b.ReadOnlyView);

        for (int j = 0; j < 6; j++)
            for (int i = 0; i < 6; i++)
                Assert.Equal(aBefore[i, j], a[i, j]);

        for (int j = 0; j < 2; j++)
            for (int i = 0; i < 6; i++)
                Assert.Equal(bBefore[i, j], b[i, j]);
    }

    // ---- the oracle and helpers ---------------------------------------------

    /// <summary>
    /// A complex operator that exposes nothing but Apply. It has no adjoint to
    /// call, so if the matrix-free overload ever needed one this would not
    /// compile.
    /// </summary>
    private sealed class ApplyOnlyComplexOperator(Matrix<Complex> a) : ILinearOperator<Complex>
    {
        public int Order => a.Rows;

        public void Apply(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                for (int i = 0; i < a.Rows; i++)
                {
                    Complex sum = Complex.Zero;
                    for (int k = 0; k < a.Columns; k++) sum += a[i, k] * x[k, j];
                    y[i, j] = sum;
                }
            }
        }
    }

    /// <summary>An operator that reports an order it has no storage for, and must never be applied.</summary>
    private sealed class HostileOrderOperator(int order) : ILinearOperator<Complex>
    {
        public int Order => order;

        public void Apply(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y) =>
            throw new InvalidOperationException("Apply must not be reached: the order should have been rejected.");
    }

    /// <summary>
    /// exp(A) by scaling and squaring a truncated Taylor series, entirely in
    /// complex arithmetic. The scaled matrix has complex 1-norm at most 1/32,
    /// so forty terms put truncation far below rounding. Deliberately naive and
    /// deliberately independent: its own matrix product, no embedding, no real
    /// algorithm, no library GEMM.
    /// </summary>
    private static Matrix<Complex> TaylorOracle(Matrix<Complex> a)
    {
        int n = a.Rows;

        int s = 0;
        double norm = ComplexOneNorm(a);
        while (norm > 0.03125)
        {
            norm *= 0.5;
            s++;
        }

        var b = Scaled(a, Math.ScaleB(1.0, -s));

        var result = Matrix.Identity<Complex>(n);
        var term = Matrix.Identity<Complex>(n);

        for (int k = 1; k <= 40; k++)
        {
            term = Scaled(Multiply(term, b), 1.0 / k);

            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    result[i, j] += term[i, j];
        }

        for (int i = 0; i < s; i++) result = Multiply(result, result);

        return result;
    }

    private static Matrix<Complex> Multiply(Matrix<Complex> a, Matrix<Complex> b)
    {
        var result = Matrix.Zeros<Complex>(a.Rows, b.Columns);

        for (int j = 0; j < b.Columns; j++)
        {
            for (int k = 0; k < a.Columns; k++)
            {
                Complex bkj = b[k, j];
                if (bkj == Complex.Zero) continue;

                for (int i = 0; i < a.Rows; i++) result[i, j] += a[i, k] * bkj;
            }
        }

        return result;
    }

    private static Matrix<Complex> ConjugateTranspose(Matrix<Complex> a)
    {
        var result = Matrix.Zeros<Complex>(a.Columns, a.Rows);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[j, i] = Complex.Conjugate(a[i, j]);

        return result;
    }

    private static Matrix<Complex> Hermitian(int n, int seed)
    {
        var m = RandomComplex(n, n, seed);
        var result = Matrix.Zeros<Complex>(n, n);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                result[i, j] = 0.5 * (m[i, j] + Complex.Conjugate(m[j, i]));

        return result;
    }

    private static Matrix<Complex> RandomComplex(int rows, int columns, int seed)
    {
        var rng = new Random(seed);
        var a = Matrix.Zeros<Complex>(rows, columns);

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                a[i, j] = new Complex((rng.NextDouble() * 2.0) - 1.0, (rng.NextDouble() * 2.0) - 1.0);

        return a;
    }

    private static Matrix<double> RandomReal(int n, int seed)
    {
        var rng = new Random(seed);
        var a = Matrix.Zeros<double>(n, n);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                a[i, j] = (rng.NextDouble() * 2.0) - 1.0;

        return a;
    }

    private static Matrix<double> WithOneNorm(Matrix<double> a, double target)
    {
        double scale = target / a.OneNorm();
        var result = Matrix.Zeros<double>(a.Rows, a.Columns);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[i, j] = a[i, j] * scale;

        return result;
    }

    private static Matrix<Complex> WithOneNorm(Matrix<Complex> a, double target) =>
        Scaled(a, target / ComplexOneNorm(a));

    private static Matrix<Complex> ToComplex(Matrix<double> a)
    {
        var result = Matrix.Zeros<Complex>(a.Rows, a.Columns);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[i, j] = a[i, j];

        return result;
    }

    private static Matrix<Complex> Scaled(Matrix<Complex> a, Complex scale)
    {
        var result = Matrix.Zeros<Complex>(a.Rows, a.Columns);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[i, j] = a[i, j] * scale;

        return result;
    }

    private static double ComplexOneNorm(Matrix<Complex> a)
    {
        double best = 0.0;

        for (int j = 0; j < a.Columns; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Rows; i++) sum += Complex.Abs(a[i, j]);
            best = Math.Max(best, sum);
        }

        return best;
    }

    private static double ColumnTwoNorm(Matrix<Complex> a, int j)
    {
        double sum = 0.0;

        for (int i = 0; i < a.Rows; i++)
        {
            double magnitude = Complex.Abs(a[i, j]);
            sum += magnitude * magnitude;
        }

        return Math.Sqrt(sum);
    }

    /// <summary>||X - Y||_1 / ||Y||_1 in the complex 1-norm, floored so an all-zero reference cannot divide by zero.</summary>
    private static double RelativeDifference(Matrix<Complex> x, Matrix<Complex> y)
    {
        double difference = 0.0;
        double reference = 0.0;

        for (int j = 0; j < x.Columns; j++)
        {
            double columnDifference = 0.0;
            double columnReference = 0.0;

            for (int i = 0; i < x.Rows; i++)
            {
                columnDifference += Complex.Abs(x[i, j] - y[i, j]);
                columnReference += Complex.Abs(y[i, j]);
            }

            difference = Math.Max(difference, columnDifference);
            reference = Math.Max(reference, columnReference);
        }

        return difference / Math.Max(reference, 1.0);
    }
}
