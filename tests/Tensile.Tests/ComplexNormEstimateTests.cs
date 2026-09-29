using System.Numerics;

namespace Tensile.Tests;

/// <summary>
/// The complex 1-norm estimator and the condition estimate built on it.
///
/// Tested the way the real estimator is (CLAUDE.md finding 8): by invariants
/// that must hold exactly, never by how often it happens to be exact. Two are
/// the real ones carried over -- it never exceeds the true norm, and with t = n
/// probes it is exact. Two are specific to the complex form. A real matrix
/// passed as complex must give exactly the real estimator's answer, because
/// every complex sign of a real number is +/-1 and every product is the real
/// product; that pins the complex path to the verified real one. And a unit
/// phase on the rows changes nothing, because it rotates Y and its signs
/// together and cancels in A^H S; that exercises complex signs that are not
/// +/-1.
/// </summary>
public class ComplexNormEstimateTests
{
    // ---- the operator --------------------------------------------------------

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void TheDenseOperatorAppliesThePowerAndItsAdjoint(int power)
    {
        const int N = 9, T = 3;

        var a = ComplexLuTests.Random(N, N, seed: 1);
        var x = ComplexLuTests.Random(N, T, seed: 2);
        var op = new ComplexDenseOperator(a, power);

        Matrix<Complex> expected = x.Clone(), expectedAdjoint = x.Clone();
        Matrix<Complex> adjointOfA = ComplexLuTests.Adjoint(a);
        for (int step = 0; step < power; step++)
        {
            expected = a.Multiply(expected.ReadOnlyView);
            expectedAdjoint = adjointOfA.Multiply(expectedAdjoint.ReadOnlyView);
        }

        var y = new Matrix<Complex>(N, T);
        op.Apply(x.ReadOnlyView, y.View);
        Assert.True(ComplexLuTests.MaxDifference(y.ReadOnlyView, expected.ReadOnlyView) < 1e-12);

        op.ApplyAdjoint(x.ReadOnlyView, y.View);
        Assert.True(ComplexLuTests.MaxDifference(y.ReadOnlyView, expectedAdjoint.ReadOnlyView) < 1e-12);
    }

    // ---- the invariants ------------------------------------------------------

    public static TheoryData<int, int, int> Ensembles
    {
        get
        {
            var data = new TheoryData<int, int, int>();
            foreach (int n in new[] { 1, 2, 5, 17, 40 })
                foreach (int t in new[] { 1, 2, 4 })
                    foreach (int power in new[] { 1, 2 })
                        data.Add(n, t, power);
            return data;
        }
    }

    /// <summary>Every probe is a genuine ||A x||_1 at ||x||_1 = 1, so no estimate can exceed the norm.</summary>
    [Theory]
    [MemberData(nameof(Ensembles))]
    public void TheEstimateNeverExceedsTheNorm(int n, int t, int power)
    {
        for (int seed = 0; seed < 10; seed++)
        {
            var a = Skewed(n, seed);
            double exact = ExactPowerNorm(a, power);
            double estimate = a.EstimateOneNorm(power, t);

            Assert.True(estimate <= exact * (1 + 1e-13), $"seed {seed}: estimate {estimate} exceeds {exact}");
            Assert.True(estimate >= exact / 3, $"seed {seed}: estimate {estimate} is far below {exact}");
        }
    }

    /// <summary>With t = n the second iteration probes every unit vector, so it sees every column.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(13)]
    public void WithAProbePerColumnTheEstimateIsExact(int n)
    {
        for (int seed = 0; seed < 5; seed++)
        {
            var a = ComplexLuTests.Random(n, n, seed);
            Assert.Equal(a.OneNorm(), a.EstimateOneNorm(columns: n), 12);
        }
    }

    /// <summary>
    /// A real matrix as complex: sign(x + 0i) is +/-1 exactly, |x + 0i| is
    /// |x| exactly, and the split products with a zero imaginary part are the
    /// real products exactly -- so the estimate must equal the real one, down
    /// to the iteration and product counts. The only thing the complex path
    /// omits is parallel-column resampling, which these inputs never trigger;
    /// with one probe column there is none to trigger in either.
    /// </summary>
    [Theory]
    [InlineData(1, 1)]
    [InlineData(6, 1)]
    [InlineData(25, 1)]
    [InlineData(25, 2)]
    [InlineData(60, 2)]
    [InlineData(60, 4)]
    public void ARealMatrixEstimatesExactlyAsTheRealEstimatorDoes(int n, int t)
    {
        for (int seed = 0; seed < 8; seed++)
        {
            var real = ComplexLuTests.RandomReal(n, n, seed);
            var complex = AsComplex(real);

            NormEstimateResult expected = NormEstimate.Of(new DenseMatrixOperator(real), t);
            NormEstimateResult actual = NormEstimate.Of(new ComplexDenseOperator(complex), t);

            Assert.Equal(expected, actual);
        }
    }

    /// <summary>
    /// D*A for a diagonal of unit phases: Y and sign(Y) both pick up D, and
    /// (DA)^H sign(DY) = A^H D^H D sign(Y) = A^H sign(Y). The probes are the
    /// same, so the estimate is -- to rounding, since the phases are not exact.
    /// This is the case where the signs are genuinely complex.
    /// </summary>
    [Fact]
    public void AUnitPhaseOnTheRowsChangesNothing()
    {
        const int N = 30;

        for (int seed = 0; seed < 6; seed++)
        {
            var a = ComplexLuTests.Random(N, N, seed);
            var rotated = a.Clone();
            var rng = new Random(seed + 100);

            for (int i = 0; i < N; i++)
            {
                Complex phase = Complex.FromPolarCoordinates(1.0, rng.NextDouble() * 2 * Math.PI);
                for (int j = 0; j < N; j++) rotated[i, j] = phase * a[i, j];
            }

            NormEstimateResult before = NormEstimate.Of(new ComplexDenseOperator(a), 2);
            NormEstimateResult after = NormEstimate.Of(new ComplexDenseOperator(rotated), 2);

            Assert.Equal(before.Value, after.Value, 12);
            Assert.Equal(before.Iterations, after.Iterations);
        }
    }

    /// <summary>A diagonal matrix: the first adjoint product points straight at the largest entry.</summary>
    [Fact]
    public void ADiagonalMatrixIsEstimatedExactly()
    {
        var a = new Matrix<Complex>(7, 7);
        for (int i = 0; i < 7; i++) a[i, i] = Complex.FromPolarCoordinates(1.0 + i % 4, i * 0.9);

        Assert.Equal(4.0, a.EstimateOneNorm(columns: 1), 14);
    }

    [Fact]
    public void AnEmptyOperatorHasZeroNorm()
    {
        NormEstimateResult result = NormEstimate.Of(new ComplexDenseOperator(new Matrix<Complex>(0, 0)));

        Assert.Equal(new NormEstimateResult(0.0, 0, 0), result);
    }

    [Fact]
    public void ANegativeOrderIsRejected() =>
        Assert.Throws<ArgumentOutOfRangeException>(() => NormEstimate.Of(new NegativeOrder()));

    [Fact]
    public void TheSurfaceRejectsWhatIsNotAnOperatorOnAMatrix()
    {
        Assert.Throws<ArgumentNullException>(() => NormEstimate.Of((IAdjointOperator<Complex>)null!));
        Assert.Throws<ArgumentException>(() => new Matrix<Complex>(3, 4).EstimateOneNorm());
        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix<Complex>(3, 3).EstimateOneNorm(power: 0));
    }

    // ---- signs ----------------------------------------------------------------

    [Fact]
    public void AComplexSignIsAUnitDirectionAndZeroPointsAlongOne()
    {
        Assert.Equal(Complex.One, ComplexKernels.Sign(Complex.Zero));
        Assert.Equal(new Complex(0.6, -0.8), ComplexKernels.Sign(new Complex(3.0, -4.0)));
        Assert.Equal(-Complex.One, ComplexKernels.Sign(new Complex(-2.5, 0.0)));

        var rng = new Random(3);
        for (int trial = 0; trial < 100; trial++)
        {
            var z = new Complex(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5);
            Assert.Equal(1.0, Complex.Abs(ComplexKernels.Sign(z)), 15);
        }

        Assert.Equal(1.0, DoubleKernels.Sign(0.0));
        Assert.Equal(1.0, DoubleKernels.Sign(-0.0));
        Assert.Equal(-1.0, DoubleKernels.Sign(-3.0));
    }

    // ---- condition --------------------------------------------------------------

    /// <summary>With a probe per column the inverse's norm is exact, so the estimate is 1/(||A||_1 ||A^-1||_1) itself.</summary>
    [Theory]
    [InlineData(3)]
    [InlineData(12)]
    [InlineData(30)]
    public void WithAProbePerColumnTheConditionEstimateIsExact(int n)
    {
        var a = ComplexLuTests.Random(n, n, seed: n);
        LuDecomposition<Complex> lu = a.FactorLu();

        double exact = 1.0 / (a.OneNorm() * lu.Solve(Matrix.Identity<Complex>(n).ReadOnlyView).OneNorm());

        Assert.Equal(exact, lu.ReciprocalCondition(columns: n), 1e-12 * exact);
    }

    /// <summary>The estimator underestimates ||A^-1||_1, so the reciprocal condition is over-estimated, never under.</summary>
    [Fact]
    public void TheDefaultConditionEstimateIsNeverBelowTheTruth()
    {
        const int N = 40;

        for (int seed = 0; seed < 10; seed++)
        {
            var a = Skewed(N, seed);
            LuDecomposition<Complex> lu = a.FactorLu();
            double exact = 1.0 / (a.OneNorm() * lu.Solve(Matrix.Identity<Complex>(N).ReadOnlyView).OneNorm());

            double estimate = lu.ReciprocalCondition();
            Assert.True(estimate >= exact * (1 - 1e-12), $"seed {seed}: {estimate} below {exact}");
            Assert.True(estimate <= exact * 3, $"seed {seed}: {estimate} far above {exact}");
        }
    }

    [Fact]
    public void ARealMatrixHasTheRealConditionNumber()
    {
        const int N = 20;

        var real = ComplexLuTests.RandomReal(N, N, seed: 4);
        double expected = real.FactorLu().ReciprocalCondition(columns: N);
        double actual = AsComplex(real).FactorLu().ReciprocalCondition(columns: N);

        Assert.Equal(expected, actual, 1e-12 * expected);
    }

    [Fact]
    public void TheConditionOfTheEdgeCases()
    {
        Assert.Equal(1.0, Matrix.Identity<Complex>(8).FactorLu().ReciprocalCondition(), 14);

        var singular = ComplexLuTests.Random(5, 5, seed: 6);
        for (int i = 0; i < 5; i++) singular[i, 2] = Complex.Zero;
        Assert.Equal(0.0, singular.FactorLu().ReciprocalCondition());

        Assert.Equal(0.0, new Matrix<Complex>(0, 0).FactorLu().ReciprocalCondition());

        Assert.Throws<InvalidOperationException>(() => ComplexLuTests.Random(4, 3, seed: 7).FactorLu().ReciprocalCondition());
    }

    /// <summary>A nearly singular matrix must be reported as such, which is what the estimate is for.</summary>
    [Fact]
    public void ANearlySingularMatrixHasATinyReciprocalCondition()
    {
        const int N = 12;

        var a = ComplexLuTests.Random(N, N, seed: 8);
        for (int i = 0; i < N; i++) a[i, 5] = a[i, 4] * new Complex(0.0, 1.0) + new Complex(1e-13, 0.0) * a[i, 7];

        Assert.True(a.FactorLu().ReciprocalCondition() < 1e-10);
    }

    [Fact]
    public void TheComplexNormsAreTheirDefinitions()
    {
        var a = ComplexLuTests.Random(6, 4, seed: 9);

        Assert.Equal(ComplexKernels.OneNorm(a.ReadOnlyView), a.OneNorm());
        Assert.Equal(ComplexKernels.InfinityNorm(a.ReadOnlyView), a.InfinityNorm());
    }

    // ---- helpers ------------------------------------------------------------------

    /// <summary>
    /// A few dominant columns, the ensemble on which the real estimator is
    /// usually but not always exact -- so both outcomes are exercised.
    /// </summary>
    private static Matrix<Complex> Skewed(int n, int seed)
    {
        var a = ComplexLuTests.Random(n, n, seed + 1000);
        var rng = new Random(seed);

        for (int j = 0; j < n; j++)
        {
            double weight = rng.NextDouble() < 0.2 ? 10.0 : 1.0;
            for (int i = 0; i < n; i++) a[i, j] *= weight;
        }

        for (int i = 0; i < n; i++) a[i, i] += n;
        return a;
    }

    private static double ExactPowerNorm(Matrix<Complex> a, int power)
    {
        Matrix<Complex> p = a.Clone();
        for (int step = 1; step < power; step++) p = a.Multiply(p.ReadOnlyView);
        return p.OneNorm();
    }

    private static Matrix<Complex> AsComplex(Matrix<double> real)
    {
        var complex = new Matrix<Complex>(real.Rows, real.Columns);
        for (int j = 0; j < real.Columns; j++)
            for (int i = 0; i < real.Rows; i++)
                complex[i, j] = real[i, j];
        return complex;
    }

    private sealed class NegativeOrder : IAdjointOperator<Complex>
    {
        public int Order => -1;

        public void Apply(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y) =>
            throw new InvalidOperationException("never called");

        public void ApplyAdjoint(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y) =>
            throw new InvalidOperationException("never called");
    }
}
