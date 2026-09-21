using Tensile.Kernels;

namespace Tensile.Tests;

/// <summary>
/// The 1-norm estimator.
///
/// A heuristic estimator cannot be tested by "is the answer right", because
/// underestimating is legitimate behaviour rather than a fault. These tests
/// pin the properties that must hold regardless:
///
/// - it never exceeds the true norm, since every probe is a genuine ||Ax||_1
///   for a unit-1-norm x;
/// - with t = n probe columns it is exact, because the second iteration probes
///   every unit vector and therefore every column;
/// - with a non-negative matrix it is exact, because the first sign matrix is
///   all ones and so A^T*S is literally the vector of column sums;
/// - it is reproducible for a fixed seed.
///
/// The first three between them exercise every piece of the bookkeeping -- the
/// sign matrix, the transposed product, the row maxima, the descending sort and
/// the unit-vector selection -- without ever asserting a specific estimate.
///
/// The estimator is driven through its public shape, an
/// <see cref="ILinearOperator"/> over a <see cref="Matrix{T}"/>, since that is
/// the only shape it has; the exact norms it is checked against are the
/// kernel layer's, over pointers, so the two sides share no code.
/// </summary>
public class NormEstimateTests
{
    public static TheoryData<int> Orders => new() { 1, 2, 3, 5, 8, 16, 17, 40, 64 };

    [Theory]
    [MemberData(nameof(Orders))]
    public void FullWidthProbeIsExact(int n)
    {
        for (int trial = 0; trial < 5; trial++)
        {
            Matrix<double> a = Random(n, seed: n * 10 + trial, stride: n + 2);

            double truth = a.OneNorm();
            double estimate = Estimate(a, columns: n);

            Assert.True(
                Math.Abs(estimate - truth) <= 1e-12 * truth,
                $"n={n} trial={trial}: estimate {estimate:E17} vs exact {truth:E17}");
        }
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void NonNegativeMatrixIsExact(int n)
    {
        for (int trial = 0; trial < 5; trial++)
        {
            Matrix<double> a = NonNegative(n, seed: n * 20 + trial);

            double truth = a.OneNorm();
            double estimate = Estimate(a);

            Assert.True(
                Math.Abs(estimate - truth) <= 1e-12 * truth,
                $"n={n} trial={trial}: estimate {estimate:E17} vs exact {truth:E17}");
        }
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void EstimateNeverExceedsTrueNorm(int n)
    {
        for (int trial = 0; trial < 10; trial++)
        {
            Matrix<double> a = Random(n, seed: n * 30 + trial);

            double truth = a.OneNorm();

            foreach (int columns in new[] { 1, 2, 4 })
            {
                double estimate = Estimate(a, columns: columns);

                Assert.True(
                    estimate <= truth * (1.0 + 1e-12),
                    $"n={n} t={columns}: estimate {estimate:E17} exceeds exact {truth:E17}");
            }
        }
    }

    [Fact]
    public void EstimateIsReproducible()
    {
        const int n = 64;

        Matrix<double> a = Random(n, seed: 99);
        var op = new DenseMatrixOperator(a);

        NormEstimateResult first = NormEstimate.Of(op);
        NormEstimateResult second = NormEstimate.Of(op);

        Assert.Equal(first, second);
    }

    /// <summary>A different seed is allowed to differ, but must stay a lower bound.</summary>
    [Fact]
    public void DifferentSeedsStayLowerBounds()
    {
        const int n = 48;

        Matrix<double> a = Random(n, seed: 101);
        double truth = a.OneNorm();

        for (int seed = 0; seed < 25; seed++)
        {
            double estimate = NormEstimate
                .Of(new DenseMatrixOperator(a), columns: 2, maxIterations: NormEstimate.DefaultMaxIterations, seed: seed)
                .Value;

            Assert.True(estimate <= truth * (1.0 + 1e-12), $"seed {seed}: {estimate:E17} > {truth:E17}");
            Assert.True(estimate >= truth * 0.4, $"seed {seed}: {estimate:E17} far below {truth:E17}");
        }
    }

    /// <summary>
    /// ||A^p||_1 without forming A^p. Checked on non-negative matrices, where
    /// the estimator is exact, so the power path is tested rather than the
    /// estimator's luck.
    /// </summary>
    [Theory]
    [InlineData(8, 2)]
    [InlineData(8, 3)]
    [InlineData(17, 2)]
    [InlineData(17, 4)]
    [InlineData(33, 3)]
    public void MatrixPowerMatchesExplicitPower(int n, int power)
    {
        Matrix<double> a = NonNegative(n, seed: n * 7 + power, scale: 1.0 / n);
        Matrix<double> accumulated = a.Clone();
        var work = new Matrix<double>(n, n);

        for (int step = 1; step < power; step++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    double sum = 0.0;
                    for (int p = 0; p < n; p++) sum += accumulated[i, p] * a[p, j];
                    work[i, j] = sum;
                }
            }

            work.View.CopyTo(accumulated.View);
        }

        double truth = accumulated.OneNorm();
        double estimate = a.EstimateOneNorm(power);

        Assert.True(
            Math.Abs(estimate - truth) <= 1e-10 * truth,
            $"n={n} power={power}: estimate {estimate:E17} vs exact {truth:E17}");
    }

    [Fact]
    public void ZeroOrderOperatorIsHandled()
    {
        NormEstimateResult result = NormEstimate.Of(new DenseMatrixOperator(new Matrix<double>(0, 0)));

        Assert.Equal(0.0, result.Value);
        Assert.Equal(0, result.Iterations);
    }

    /// <summary>
    /// The operator's own validation: panels whose rows disagree with the
    /// order, or whose widths disagree with each other, are argument errors
    /// and never reach the kernel.
    /// </summary>
    [Fact]
    public void DenseOperatorRejectsMismatchedPanels()
    {
        var op = new DenseMatrixOperator(Matrix.Identity<double>(4));

        Assert.Throws<ArgumentException>(() => op.Apply(new Matrix<double>(3, 2), new Matrix<double>(4, 2)));
        Assert.Throws<ArgumentException>(() => op.Apply(new Matrix<double>(4, 2), new Matrix<double>(4, 3)));
        Assert.Throws<ArgumentException>(() => op.ApplyTranspose(new Matrix<double>(4, 2), new Matrix<double>(5, 2)));
        Assert.Throws<ArgumentException>(() => new DenseMatrixOperator(new Matrix<double>(3, 4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DenseMatrixOperator(Matrix.Identity<double>(4), power: 0));
    }

    /// <summary>
    /// Every product the estimator asks for stays within the panels it hands
    /// out. An operator that records what it was given is the cheapest way to
    /// see the estimator's side of the contract.
    /// </summary>
    [Fact]
    public void EstimatorHandsOperatorConsistentPanels()
    {
        const int n = 12;
        var recorder = new RecordingOperator(Random(n, seed: 5));

        NormEstimate.Of(recorder, columns: 3);

        Assert.NotEmpty(recorder.Shapes);
        Assert.All(recorder.Shapes, shape => Assert.Equal((n, 3, n, 3), shape));
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public unsafe void OneNormMatchesNaiveColumnSums(int n)
    {
        using var a = TestMatrix.Random(n, n, seed: n + 555, stride: n + 4);

        double expected = 0.0;

        for (int j = 0; j < n; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += Math.Abs(a[i, j]);
            expected = Math.Max(expected, sum);
        }

        Assert.Equal(expected, Norms.One(n, n, a.Data, a.Stride), 12);
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public unsafe void InfinityNormMatchesNaiveRowSums(int n)
    {
        using var a = TestMatrix.Random(n, n, seed: n + 666, stride: n + 4);

        double expected = 0.0;

        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += Math.Abs(a[i, j]);
            expected = Math.Max(expected, sum);
        }

        Assert.Equal(expected, Norms.Infinity(n, n, a.Data, a.Stride), 12);
    }

    private static double Estimate(Matrix<double> a, int columns = NormEstimate.DefaultColumns) =>
        NormEstimate.Of(new DenseMatrixOperator(a), columns).Value;

    /// <summary>Uniform random entries in [-0.5, 0.5], with an optional stride so padding is exercised.</summary>
    private static Matrix<double> Random(int n, int seed, int stride = 0)
    {
        var matrix = new Matrix<double>(n, n, stride);
        var rng = new Random(seed);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                matrix[i, j] = rng.NextDouble() - 0.5;

        return matrix;
    }

    private static Matrix<double> NonNegative(int n, int seed, double scale = 1.0)
    {
        var matrix = new Matrix<double>(n, n, stride: n + 2);
        var rng = new Random(seed);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                matrix[i, j] = scale * (0.05 + rng.NextDouble());

        return matrix;
    }

    private sealed class RecordingOperator(Matrix<double> a) : ITransposableOperator
    {
        private readonly DenseMatrixOperator _inner = new(a);

        public List<(int, int, int, int)> Shapes { get; } = [];

        public int Order => _inner.Order;

        public void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y)
        {
            Shapes.Add((x.Rows, x.Columns, y.Rows, y.Columns));
            _inner.Apply(x, y);
        }

        public void ApplyTranspose(ReadOnlyMatrixView<double> x, MatrixView<double> y)
        {
            Shapes.Add((x.Rows, x.Columns, y.Rows, y.Columns));
            _inner.ApplyTranspose(x, y);
        }
    }
}

/// <summary>
/// Condition estimation, which is the norm estimator applied to A^-1 through
/// the LU factors. Generic over the kernel because the trailing update of the
/// factorization is a GEMM, and a wrong kernel there would surface here.
/// </summary>
public abstract class ConditionContract<TCase> where TCase : struct, IKernelCase
{
    /// <summary>The kernel this instantiation of the contract runs against.</summary>
    internal static readonly KernelDriver Kernel = KernelDriver.For<TCase>();

    private readonly Workspace _workspace;

    protected ConditionContract()
    {
        Assert.SkipUnless(Kernel.IsSupported, $"{Kernel.Name} is not supported on this CPU");
        _workspace = Kernel.Workspace(multithreaded: false);
    }

    [Fact]
    public void IdentityIsPerfectlyConditioned()
    {
        LuDecomposition lu = Matrix.Identity<double>(40).FactorLu(blockSize: 8, _workspace);

        Assert.Equal(1.0, lu.ReciprocalCondition(), 12);
    }

    /// <summary>
    /// Against the exact reciprocal condition number, computed by forming A^-1
    /// explicitly -- which is affordable here only because n is small, and is
    /// exactly what the estimator exists to avoid.
    ///
    /// The estimate must be an OVER-estimate of rcond, since it underestimates
    /// ||A^-1||_1. With t = n it must be exact.
    /// </summary>
    [Theory]
    [InlineData(5)]
    [InlineData(16)]
    [InlineData(33)]
    [InlineData(48)]
    public void TracksTheExactReciprocalCondition(int n)
    {
        Matrix<double> a = RandomDiagonallyDominant(n, seed: n * 3);
        LuDecomposition lu = a.FactorLu(blockSize: 8, _workspace);

        // A^-1 by solving against the identity.
        Matrix<double> inverse = lu.Solve(Matrix.Identity<double>(n));

        double exact = 1.0 / (a.OneNorm() * inverse.OneNorm());

        double estimated = lu.ReciprocalCondition();
        double full = lu.ReciprocalCondition(columns: n);

        Assert.True(estimated >= exact * (1.0 - 1e-10),
            $"n={n}: rcond estimate {estimated:E6} is below the exact {exact:E6}");
        Assert.True(estimated <= exact * 3.0,
            $"n={n}: rcond estimate {estimated:E6} is far above the exact {exact:E6}");
        Assert.True(Math.Abs(full - exact) <= 1e-9 * exact,
            $"n={n}: full-width rcond {full:E17} should equal the exact {exact:E17}");
    }

    /// <summary>Hilbert matrices are the standard ill-conditioned family.</summary>
    [Fact]
    public void HilbertConditioningDegradesWithOrder()
    {
        double previous = double.PositiveInfinity;

        foreach (int n in new[] { 4, 6, 8, 10, 12 })
        {
            var a = new Matrix<double>(n, n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    a[i, j] = 1.0 / (i + j + 1);

            double rcond = a.FactorLu(blockSize: 4, _workspace).ReciprocalCondition();

            Assert.True(rcond < previous, $"rcond at n={n} ({rcond:E3}) did not fall below {previous:E3}");
            previous = rcond;
        }

        Assert.True(previous < 1e-12, $"Hilbert(12) rcond {previous:E3} is implausibly healthy");
    }

    [Fact]
    public void SingularMatrixGivesZero()
    {
        const int n = 12;

        Matrix<double> a = RandomDiagonallyDominant(n, seed: 77);
        for (int i = 0; i < n; i++) a[i, 4] = 0.0;

        LuDecomposition lu = a.FactorLu(blockSize: 4, _workspace);

        Assert.True(lu.IsSingular);
        Assert.Equal(0.0, lu.ReciprocalCondition());
    }

    private static Matrix<double> RandomDiagonallyDominant(int n, int seed)
    {
        var matrix = new Matrix<double>(n, n);
        var rng = new Random(seed);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                matrix[i, j] = rng.NextDouble() - 0.5;

        for (int i = 0; i < n; i++) matrix[i, i] += n;
        return matrix;
    }
}

public sealed class ScalarConditionTests : ConditionContract<ScalarCase>;
public sealed class Avx2ConditionTests : ConditionContract<Avx2Case>;
public sealed class Avx512ConditionTests : ConditionContract<Avx512Case>;
