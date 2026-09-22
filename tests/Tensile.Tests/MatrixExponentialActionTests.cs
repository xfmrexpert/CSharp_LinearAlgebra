namespace Tensile.Tests;

/// <summary>
/// The action of the matrix exponential.
///
/// This algorithm gets a better oracle than <c>expm</c> did: <c>expm</c> is
/// already in the library and already verified against an independent Taylor
/// series, so <c>Expm(A) * B</c> is a trusted reference computed by a
/// genuinely different route -- Padé approximants and a linear solve, against
/// this one's scaled Taylor recurrence. Where they agree, two unrelated
/// algorithms agree.
///
/// What that does not cover is the parameter selection, which can be badly
/// wrong while the answer stays right, because the early-termination test
/// rescues an m that is too large and a large s merely wastes work. So the
/// degree and scaling are observed directly through <c>ExpmvDiagnostics</c>,
/// and the tests below assert that the selection responds to the input rather
/// than sitting at a constant.
/// </summary>
public class MatrixExponentialActionTests
{
    private const double Tolerance = 1e-10;

    // ---- against expm, the trusted oracle -----------------------------------

    [Theory]
    [InlineData(8, 0.01, 1.0)]
    [InlineData(8, 0.5, 1.0)]
    [InlineData(8, 4.0, 1.0)]
    [InlineData(8, 30.0, 1.0)]
    [InlineData(16, 1.0, 1.0)]
    [InlineData(16, 12.0, 1.0)]
    [InlineData(32, 0.2, 1.0)]
    [InlineData(32, 8.0, 1.0)]
    [InlineData(48, 25.0, 1.0)]
    public void AgreesWithExpmTimesB(int n, double norm, double t)
    {
        var a = RandomWithOneNorm(n, norm, seed: (n * 31) + (int)(norm * 7));
        var b = RandomPanel(n, columns: 3, seed: 5150);

        var actual = a.Expmv(b.ReadOnlyView, t);
        var expected = a.Expm().Multiply(b.ReadOnlyView);

        Assert.True(
            RelativeDifference(actual, expected) <= Tolerance,
            $"n={n}, ||A||={norm}, t={t}: {RelativeDifference(actual, expected):E3}");
    }

    /// <summary>
    /// The time enters the algorithm twice — through the parameter selection
    /// and through the recurrence's coefficients — so a sign or scale error
    /// there survives the t = 1 cases above.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(1e-3)]
    [InlineData(0.25)]
    [InlineData(2.5)]
    [InlineData(-1.0)]
    [InlineData(-7.5)]
    public void HandlesTheTimeArgument(double t)
    {
        const int N = 12;

        var a = RandomWithOneNorm(N, 3.0, seed: 6060);
        var b = RandomPanel(N, columns: 2, seed: 7070);

        var actual = a.Expmv(b.ReadOnlyView, t);
        var expected = Scaled(a, t).Expm().Multiply(b.ReadOnlyView);

        Assert.True(
            RelativeDifference(actual, expected) <= Tolerance,
            $"t={t}: {RelativeDifference(actual, expected):E3}");
    }

    /// <summary>
    /// A single right-hand side is the case that motivates the whole
    /// algorithm, and it is a different shape through every loop.
    /// </summary>
    [Theory]
    [InlineData(1.0)]
    [InlineData(20.0)]
    public void HandlesASingleVector(double norm)
    {
        const int N = 20;

        var a = RandomWithOneNorm(N, norm, seed: 8080);
        var b = RandomPanel(N, columns: 1, seed: 9090);

        var actual = a.Expmv(b.ReadOnlyView);
        var expected = a.Expm().Multiply(b.ReadOnlyView);

        Assert.Equal(1, actual.Columns);
        Assert.True(RelativeDifference(actual, expected) <= Tolerance);
    }

    /// <summary>
    /// A strictly triangular matrix is nilpotent, so every ||A^p||^(1/p) is
    /// far below ||A||. This is the case the sharp parameter selection exists
    /// for, and where using ||A|| alone would overscale.
    /// </summary>
    [Theory]
    [InlineData(12, 2.0)]
    [InlineData(12, 20.0)]
    [InlineData(24, 6.0)]
    public void AgreesWithExpmOnANilpotentMatrix(int n, double scale)
    {
        var a = StrictlyUpperTriangular(n, scale, seed: 1717);
        var b = RandomPanel(n, columns: 2, seed: 1818);

        var actual = a.Expmv(b.ReadOnlyView);
        var expected = a.Expm().Multiply(b.ReadOnlyView);

        Assert.True(
            RelativeDifference(actual, expected) <= Tolerance,
            $"n={n}, scale={scale}: {RelativeDifference(actual, expected):E3}");
    }

    // ---- closed forms -------------------------------------------------------

    [Fact]
    public void ZeroOperatorLeavesBAlone()
    {
        const int N = 6;

        var b = RandomPanel(N, columns: 2, seed: 2121);
        var result = Matrix.Zeros<double>(N, N).Expmv(b.ReadOnlyView);

        for (int j = 0; j < 2; j++)
            for (int i = 0; i < N; i++)
                Assert.Equal(b[i, j], result[i, j], 14);
    }

    /// <summary>exp(tD)b is element-wise for a diagonal D, whatever the scaling picks.</summary>
    [Theory]
    [InlineData(0.5)]
    [InlineData(9.0)]
    public void DiagonalActsElementWise(double scale)
    {
        double[] diagonal = [0.0, 1.0, -1.0, 0.25, -2.0, 0.75];
        int n = diagonal.Length;

        var a = Matrix.Zeros<double>(n, n);
        for (int i = 0; i < n; i++) a[i, i] = diagonal[i] * scale;

        var b = RandomPanel(n, columns: 1, seed: 2323);
        var result = a.Expmv(b.ReadOnlyView);

        for (int i = 0; i < n; i++)
        {
            double expected = Math.Exp(diagonal[i] * scale) * b[i, 0];
            Assert.True(
                Math.Abs(result[i, 0] - expected) <= 1e-11 * Math.Max(Math.Abs(expected), 1.0),
                $"[{i}] {result[i, 0]:E17} vs {expected:E17}");
        }
    }

    /// <summary>
    /// exp((t1+t2)A)b = exp(t1 A) exp(t2 A) b. The two sides choose different
    /// degrees and scalings, so this checks the parameter selection is
    /// self-consistent and not merely repeatable.
    /// </summary>
    [Fact]
    public void CompositionInTimeHolds()
    {
        const int N = 14;

        var a = RandomWithOneNorm(N, 5.0, seed: 2525);
        var b = RandomPanel(N, columns: 2, seed: 2626);

        var whole = a.Expmv(b.ReadOnlyView, 1.5);
        var split = a.Expmv(a.Expmv(b.ReadOnlyView, 0.5).ReadOnlyView, 1.0);

        Assert.True(RelativeDifference(whole, split) <= Tolerance);
    }

    // ---- the matrix-free path -----------------------------------------------

    /// <summary>
    /// The operator overload must reach the same answer as the dense one. It
    /// takes a different route to the parameters — no trace shift, no sharp
    /// estimates — so agreement here says the fallback is conservative rather
    /// than wrong.
    /// </summary>
    [Theory]
    [InlineData(10, 0.5)]
    [InlineData(10, 6.0)]
    [InlineData(24, 15.0)]
    public void OperatorPathAgreesWithTheDensePath(int n, double norm)
    {
        var a = RandomWithOneNorm(n, norm, seed: 3131 + n);
        var b = RandomPanel(n, columns: 2, seed: 3232);

        var dense = a.Expmv(b.ReadOnlyView);
        var free = MatrixExponentialAction.Expmv(
            new ApplyOnlyOperator(a), b.ReadOnlyView, t: 1.0, oneNormBound: a.OneNorm());

        Assert.True(
            RelativeDifference(dense, free) <= Tolerance,
            $"n={n}, ||A||={norm}: {RelativeDifference(dense, free):E3}");
    }

    /// <summary>
    /// The operator overload must never touch the transpose. An operator whose
    /// ApplyTranspose throws proves it, and this is the contract that lets a
    /// matrix-free FEM operator use this at all.
    /// </summary>
    [Fact]
    public void OperatorPathNeverAppliesTheTranspose()
    {
        const int N = 10;

        var a = RandomWithOneNorm(N, 4.0, seed: 3333);
        var b = RandomPanel(N, columns: 1, seed: 3434);

        // ApplyOnlyOperator implements only ILinearOperator, so there is no
        // transpose to call even by accident; this asserts the overload binds
        // to the narrow interface and completes.
        var result = MatrixExponentialAction.Expmv(
            new ApplyOnlyOperator(a), b.ReadOnlyView, t: 1.0, oneNormBound: a.OneNorm());

        var expected = a.Expm().Multiply(b.ReadOnlyView);
        Assert.True(RelativeDifference(result, expected) <= Tolerance);
    }

    /// <summary>A bound larger than the true norm must still be correct, only slower.</summary>
    [Fact]
    public void AnOverstatedNormBoundIsStillCorrect()
    {
        const int N = 10;

        var a = RandomWithOneNorm(N, 2.0, seed: 3535);
        var b = RandomPanel(N, columns: 1, seed: 3636);

        var tight = MatrixExponentialAction.Expmv(
            new ApplyOnlyOperator(a), b.ReadOnlyView, 1.0, oneNormBound: a.OneNorm());
        var loose = MatrixExponentialAction.Expmv(
            new ApplyOnlyOperator(a), b.ReadOnlyView, 1.0, oneNormBound: a.OneNorm() * 50.0);

        Assert.True(RelativeDifference(tight, loose) <= Tolerance);
    }

    // ---- parameter selection ------------------------------------------------

    /// <summary>
    /// The selection must respond to the norm. A degree and scaling that never
    /// moved would mean the search had collapsed to a constant, and every
    /// accuracy test above would still pass because a too-large m is rescued
    /// by early termination and a too-large s only wastes work.
    /// </summary>
    [Fact]
    public void ScalingGrowsWithTheNorm()
    {
        var scalings = new List<int>();
        var degrees = new HashSet<int>();

        foreach (double norm in new[] { 0.01, 1.0, 10.0, 100.0, 1000.0 })
        {
            var a = RandomWithOneNorm(10, norm, seed: 4040);
            var b = RandomPanel(10, columns: 1, seed: 4141);

            var diagnostics = new ExpmvDiagnostics();
            _ = MatrixExponentialAction.Expmv(a, b.ReadOnlyView, 1.0, diagnostics);

            scalings.Add(diagnostics.Scaling);
            degrees.Add(diagnostics.Degree);
        }

        // Monotone non-decreasing, and genuinely spread rather than constant.
        for (int i = 1; i < scalings.Count; i++)
            Assert.True(scalings[i] >= scalings[i - 1], $"scalings not monotone: {string.Join(", ", scalings)}");

        Assert.True(scalings[^1] > scalings[0] * 10, $"scaling barely moved: {string.Join(", ", scalings)}");
        Assert.True(degrees.Count > 1, $"degree never varied: {string.Join(", ", degrees)}");
    }

    /// <summary>
    /// Early termination is the difference between this and a fixed-degree
    /// Taylor series, and it must actually fire on ordinary input.
    /// </summary>
    [Fact]
    public void TheInnerLoopTerminatesEarly()
    {
        var a = RandomWithOneNorm(16, 6.0, seed: 4242);
        var b = RandomPanel(16, columns: 1, seed: 4343);

        var diagnostics = new ExpmvDiagnostics();
        _ = MatrixExponentialAction.Expmv(a, b.ReadOnlyView, 1.0, diagnostics);

        Assert.True(diagnostics.EarlyExits > 0, "the truncation test never fired");
        Assert.True(
            diagnostics.Applications < diagnostics.Degree * diagnostics.Scaling,
            $"{diagnostics.Applications} applications against a cap of {diagnostics.Degree * diagnostics.Scaling}");
    }

    /// <summary>
    /// The sharp bound is what the dense path buys over the matrix-free one.
    /// On a nilpotent matrix, where ||A^p||^(1/p) collapses but ||A|| does
    /// not, the dense path must do strictly less work.
    /// </summary>
    [Fact]
    public void TheSharpBoundBeatsTheNormBoundOnANilpotentMatrix()
    {
        const int N = 16;

        var a = StrictlyUpperTriangular(N, 12.0, seed: 4444);
        var b = RandomPanel(N, columns: 1, seed: 4545);

        var dense = new ExpmvDiagnostics();
        _ = MatrixExponentialAction.Expmv(a, b.ReadOnlyView, 1.0, dense);

        var free = new ExpmvDiagnostics();
        _ = MatrixExponentialAction.Expmv(
            new ApplyOnlyOperator(a), b.ReadOnlyView, 1.0, a.OneNorm(), free);

        Assert.True(
            dense.Applications < free.Applications,
            $"sharp bound used {dense.Applications} applications, norm bound {free.Applications}");
    }

    /// <summary>A zero operator needs no work at all beyond the shift's scalar.</summary>
    [Fact]
    public void AZeroOperatorCostsNoApplications()
    {
        var b = RandomPanel(8, columns: 1, seed: 4646);

        var diagnostics = new ExpmvDiagnostics();
        _ = MatrixExponentialAction.Expmv(Matrix.Zeros<double>(8, 8), b.ReadOnlyView, 1.0, diagnostics);

        Assert.Equal(0, diagnostics.Degree);
        Assert.Equal(1, diagnostics.Scaling);
        Assert.Equal(0, diagnostics.Applications);
    }

    // ---- degenerate shapes and arguments ------------------------------------

    [Fact]
    public void EmptyOperatorReturnsBUnchanged()
    {
        var b = Matrix.Zeros<double>(0, 0);
        var result = Matrix.Zeros<double>(0, 0).Expmv(b.ReadOnlyView);

        Assert.Equal(0, result.Rows);
        Assert.Equal(0, result.Columns);
    }

    [Fact]
    public void NoColumnsReturnsBUnchanged()
    {
        var a = Matrix.Identity<double>(4);
        var b = Matrix.Zeros<double>(4, 0);

        var result = a.Expmv(b.ReadOnlyView);

        Assert.Equal(4, result.Rows);
        Assert.Equal(0, result.Columns);
    }

    [Fact]
    public void RectangularOperatorIsRejected()
    {
        var a = Matrix.Zeros<double>(3, 4);
        var b = Matrix.Zeros<double>(3, 1);

        var error = Assert.Throws<ArgumentException>(() => a.Expmv(b.ReadOnlyView));
        Assert.Contains("square", error.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void MismatchedRowsAreRejected()
    {
        var a = Matrix.Identity<double>(4);
        var b = Matrix.Zeros<double>(5, 1);

        Assert.Throws<ArgumentException>(() => a.Expmv(b.ReadOnlyView));
    }

    [Fact]
    public void NullOperatorIsRejected()
    {
        Matrix<double>? a = null;
        var b = Matrix.Zeros<double>(4, 1);

        Assert.Throws<ArgumentNullException>(() => a!.Expmv(b.ReadOnlyView));
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AnUnusableNormBoundIsRejected(double bound)
    {
        var a = Matrix.Identity<double>(4);
        var b = Matrix.Zeros<double>(4, 1);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => MatrixExponentialAction.Expmv(new ApplyOnlyOperator(a), b.ReadOnlyView, 1.0, bound));
    }

    /// <summary>Neither the operator nor B may be modified.</summary>
    [Fact]
    public void TheInputsAreNotModified()
    {
        var a = RandomWithOneNorm(8, 4.0, seed: 4747);
        var b = RandomPanel(8, columns: 2, seed: 4848);

        var aBefore = a.Clone();
        var bBefore = b.Clone();

        _ = a.Expmv(b.ReadOnlyView);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                Assert.Equal(aBefore[i, j], a[i, j]);

        for (int j = 0; j < b.Columns; j++)
            for (int i = 0; i < b.Rows; i++)
                Assert.Equal(bBefore[i, j], b[i, j]);
    }

    // ---- helpers ------------------------------------------------------------

    /// <summary>
    /// An operator that implements only <see cref="ILinearOperator"/>. It has
    /// no transpose to apply, which is the whole point: if the matrix-free
    /// overload ever needed one, this would not compile.
    /// </summary>
    private sealed class ApplyOnlyOperator(Matrix<double> a) : ILinearOperator
    {
        public int Order => a.Rows;

        public void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y)
        {
            for (int j = 0; j < x.Columns; j++)
            {
                for (int i = 0; i < a.Rows; i++)
                {
                    double sum = 0.0;
                    for (int k = 0; k < a.Columns; k++) sum += a[i, k] * x[k, j];
                    y[i, j] = sum;
                }
            }
        }
    }

    private static Matrix<double> RandomWithOneNorm(int n, double target, int seed)
    {
        var a = RandomPanel(n, n, seed);
        return Scaled(a, target / a.OneNorm());
    }

    private static Matrix<double> RandomPanel(int rows, int columns, int seed)
    {
        var rng = new Random(seed);
        var a = Matrix.Zeros<double>(rows, columns);

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                a[i, j] = (rng.NextDouble() * 2.0) - 1.0;

        return a;
    }

    private static Matrix<double> StrictlyUpperTriangular(int n, double scale, int seed)
    {
        var rng = new Random(seed);
        var a = Matrix.Zeros<double>(n, n);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < j; i++)
                a[i, j] = ((rng.NextDouble() * 2.0) - 1.0) * scale;

        return a;
    }

    private static Matrix<double> Scaled(Matrix<double> a, double scale)
    {
        var result = Matrix.Zeros<double>(a.Rows, a.Columns);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[i, j] = a[i, j] * scale;

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
}
