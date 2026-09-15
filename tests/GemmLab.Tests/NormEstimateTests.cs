namespace GemmLab.Tests;

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
/// </summary>
public unsafe class NormEstimateTests
{
    public static TheoryData<int> Orders => new() { 1, 2, 3, 5, 8, 16, 17, 40, 64 };

    [Theory]
    [MemberData(nameof(Orders))]
    public void FullWidthProbeIsExact(int n)
    {
        for (int trial = 0; trial < 5; trial++)
        {
            using var a = Matrix.Random(n, n, seed: n * 10 + trial, stride: n + 2);

            double truth = Norms.One(n, n, a.Data, a.Stride);
            double estimate = NormEstimate.OfMatrix(n, a.Data, a.Stride, power: 1, columns: n).Value;

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
            using var a = NonNegative(n, seed: n * 20 + trial);

            double truth = Norms.One(n, n, a.Data, a.Stride);
            double estimate = NormEstimate.OfMatrix(n, a.Data, a.Stride).Value;

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
            using var a = Matrix.Random(n, n, seed: n * 30 + trial);

            double truth = Norms.One(n, n, a.Data, a.Stride);

            foreach (int columns in new[] { 1, 2, 4 })
            {
                double estimate = NormEstimate.OfMatrix(n, a.Data, a.Stride, power: 1, columns: columns).Value;

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

        using var a = Matrix.Random(n, n, seed: 99);

        var first = NormEstimate.OfMatrix(n, a.Data, a.Stride);
        var second = NormEstimate.OfMatrix(n, a.Data, a.Stride);

        Assert.Equal(first, second);
    }

    /// <summary>A different seed is allowed to differ, but must stay a lower bound.</summary>
    [Fact]
    public void DifferentSeedsStayLowerBounds()
    {
        const int n = 48;

        using var a = Matrix.Random(n, n, seed: 101);
        double truth = Norms.One(n, n, a.Data, a.Stride);

        for (int seed = 0; seed < 25; seed++)
        {
            double estimate = NormEstimate
                .OfMatrix(n, a.Data, a.Stride, power: 1, columns: 2,
                    maxIterations: NormEstimate.DefaultMaxIterations, seed: seed)
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
        using var a = NonNegative(n, seed: n * 7 + power, scale: 1.0 / n);
        using var accumulated = a.Clone();
        using var work = new Matrix(n, n);

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

            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    accumulated[i, j] = work[i, j];
        }

        double truth = Norms.One(n, n, accumulated.Data, accumulated.Stride);
        double estimate = NormEstimate.OfMatrix(n, a.Data, a.Stride, power).Value;

        Assert.True(
            Math.Abs(estimate - truth) <= 1e-10 * truth,
            $"n={n} power={power}: estimate {estimate:E17} vs exact {truth:E17}");
    }

    [Fact]
    public void ZeroOrderOperatorIsHandled()
    {
        var result = NormEstimate.OfMatrix(0, null, 1);

        Assert.Equal(0.0, result.Value);
        Assert.Equal(0, result.Iterations);
    }

    [Theory]
    [MemberData(nameof(Orders))]
    public void OneNormMatchesNaiveColumnSums(int n)
    {
        using var a = Matrix.Random(n, n, seed: n + 555, stride: n + 4);

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
    public void InfinityNormMatchesNaiveRowSums(int n)
    {
        using var a = Matrix.Random(n, n, seed: n + 666, stride: n + 4);

        double expected = 0.0;

        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++) sum += Math.Abs(a[i, j]);
            expected = Math.Max(expected, sum);
        }

        Assert.Equal(expected, Norms.Infinity(n, n, a.Data, a.Stride), 12);
    }

    private static Matrix NonNegative(int n, int seed, double scale = 1.0)
    {
        var matrix = new Matrix(n, n, stride: n + 2);
        var rng = new Random(seed);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                matrix[i, j] = scale * (0.05 + rng.NextDouble());

        return matrix;
    }
}

/// <summary>
/// Condition estimation, which is the norm estimator applied to A^-1 through
/// the LU factors.
/// </summary>
public abstract unsafe class ConditionContract<TKernel> where TKernel : struct, IMicroKernel
{
    protected ConditionContract() =>
        Assert.SkipUnless(TKernel.IsSupported, $"{TKernel.Name} is not supported on this CPU");

    [Fact]
    public void IdentityIsPerfectlyConditioned()
    {
        const int n = 40;

        using var a = new Matrix(n, n);
        for (int i = 0; i < n; i++) a[i, i] = 1.0;

        double norm = Norms.One(n, n, a.Data, a.Stride);

        using var gemm = GemmDispatch.Serial<TKernel>();
        using var lu = Lu.Factor<TKernel>(n, n, a.Data, a.Stride, gemm, 8);

        Assert.Equal(1.0, Condition.ReciprocalOne(norm, lu), 12);
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
        using var original = Matrix.RandomDiagonallyDominant(n, seed: n * 3);
        using var factors = original.Clone();

        using var gemm = GemmDispatch.Serial<TKernel>();
        using var lu = Lu.Factor<TKernel>(n, n, factors.Data, factors.Stride, gemm, 8);

        double normOfA = Norms.One(n, n, original.Data, original.Stride);

        // A^-1 by solving against the identity.
        using var inverse = new Matrix(n, n);
        for (int i = 0; i < n; i++) inverse[i, i] = 1.0;
        Lu.Solve(lu, n, inverse.Data, inverse.Stride);

        double exact = 1.0 / (normOfA * Norms.One(n, n, inverse.Data, inverse.Stride));

        double estimated = Condition.ReciprocalOne(normOfA, lu);
        double full = Condition.ReciprocalOne(normOfA, lu, columns: n);

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
            using var a = new Matrix(n, n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    a[i, j] = 1.0 / (i + j + 1);

            double norm = Norms.One(n, n, a.Data, a.Stride);

            using var gemm = GemmDispatch.Serial<TKernel>();
            using var lu = Lu.Factor<TKernel>(n, n, a.Data, a.Stride, gemm, 4);

            double rcond = Condition.ReciprocalOne(norm, lu);

            Assert.True(rcond < previous, $"rcond at n={n} ({rcond:E3}) did not fall below {previous:E3}");
            previous = rcond;
        }

        Assert.True(previous < 1e-12, $"Hilbert(12) rcond {previous:E3} is implausibly healthy");
    }

    [Fact]
    public void SingularMatrixGivesZero()
    {
        const int n = 12;

        using var a = Matrix.RandomDiagonallyDominant(n, seed: 77);
        for (int i = 0; i < n; i++) a[i, 4] = 0.0;

        double norm = Norms.One(n, n, a.Data, a.Stride);

        using var gemm = GemmDispatch.Serial<TKernel>();
        using var lu = Lu.Factor<TKernel>(n, n, a.Data, a.Stride, gemm, 4);

        Assert.Equal(0.0, Condition.ReciprocalOne(norm, lu));
    }
}

public sealed class ScalarConditionTests : ConditionContract<ScalarKernel4x4>;
public sealed class Avx2ConditionTests : ConditionContract<Avx2Kernel8x6>;
public sealed class Avx512ConditionTests : ConditionContract<Avx512Kernel16x8>;
