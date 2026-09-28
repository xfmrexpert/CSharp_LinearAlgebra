using System.Numerics;

namespace Tensile.Tests;

/// <summary>
/// Complex products by the 4M method, checked per micro-kernel.
///
/// The accuracy assertion is componentwise and derived, not a tuned tolerance.
/// Each entry of a conventional complex product has rounding error bounded by
/// about k*u*(|A||B|)_ij, and 4M's bound is the same form -- that is the claim
/// that justifies it over 3M -- so the difference between 4M and a naive
/// complex reference must be within twice that, with some slack. The bound
/// uses |a||b| >= |Re a||Re b| + |Im a||Im b|, so it covers both parts.
///
/// Every shape runs on every micro-kernel the host supports, serial and
/// threaded, because 4M is four calls into whichever GEMM the workspace owns.
/// </summary>
public abstract class ComplexMultiplyContract<TCase> where TCase : struct, IKernelCase
{
    internal static readonly KernelDriver Kernel = KernelDriver.For<TCase>();

    protected ComplexMultiplyContract() =>
        Assert.SkipUnless(Kernel.IsSupported, $"{Kernel.Name} is not supported on this CPU");

    /// <summary>Shapes straddling the micro-tile edges in every dimension, as in the real contract.</summary>
    public static TheoryData<int, int, int> Shapes => new()
    {
        { 1, 1, 1 },
        { 1, 1, 64 },
        { 8, 6, 4 },
        { 17, 9, 5 },
        { 31, 33, 7 },
        { 64, 64, 64 },
        { 65, 63, 66 },
        { 100, 37, 129 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SerialMatchesReference(int m, int n, int k) =>
        Check(m, n, k, Complex.One, Complex.Zero, multithreaded: false);

    [Theory]
    [MemberData(nameof(Shapes))]
    public void ThreadedMatchesReference(int m, int n, int k) =>
        Check(m, n, k, Complex.One, Complex.Zero, multithreaded: true);

    /// <summary>
    /// The scalar forms, including complex alpha and beta, which mix the real
    /// and imaginary products and so exercise Combine's general branch rather
    /// than its special cases.
    /// </summary>
    [Theory]
    [InlineData(1.0, 0.0, 0.0, 0.0)]
    [InlineData(1.0, 0.0, 1.0, 0.0)]
    [InlineData(-1.0, 0.0, 1.0, 0.0)]
    [InlineData(0.0, 0.0, 1.0, 0.0)]
    [InlineData(0.5, -2.0, 0.0, 0.0)]
    [InlineData(0.5, -2.0, 1.0, 1.0)]
    [InlineData(2.0, 0.0, -0.5, 0.75)]
    public void ScalarsAreApplied(double alphaRe, double alphaIm, double betaRe, double betaIm) =>
        Check(23, 19, 17, new Complex(alphaRe, alphaIm), new Complex(betaRe, betaIm), multithreaded: false);

    /// <summary>Views with a padded stride, so no operand is contiguous.</summary>
    [Fact]
    public void StridedViewsAreHonoured()
    {
        using var workspace = Kernel.Workspace(multithreaded: false);

        var aStore = new Matrix<Complex>(20, 11, stride: 27);
        var bStore = new Matrix<Complex>(11, 13, stride: 16);
        var cStore = new Matrix<Complex>(20, 13, stride: 25);

        Fill(aStore, seed: 3);
        Fill(bStore, seed: 4);

        workspace.Multiply(aStore.ReadOnlyView, bStore.ReadOnlyView, cStore.View);

        AssertWithinBound(cStore, aStore, bStore, Reference(aStore, bStore, null, Complex.One, Complex.Zero));
    }

    private static void Check(int m, int n, int k, Complex alpha, Complex beta, bool multithreaded)
    {
        using var workspace = Kernel.Workspace(multithreaded);

        var a = new Matrix<Complex>(m, k);
        var b = new Matrix<Complex>(k, n);
        var c = new Matrix<Complex>(m, n);

        Fill(a, seed: (m * 31) + k);
        Fill(b, seed: (k * 17) + n);
        Fill(c, seed: (m * 7) + n);

        var expected = Reference(a, b, c, alpha, beta);

        workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View, alpha, beta);

        AssertWithinBound(c, a, b, expected, alpha, beta);
    }

    /// <summary>beta*C + alpha*A*B by the textbook triple loop in complex arithmetic.</summary>
    internal static Matrix<Complex> Reference(Matrix<Complex> a, Matrix<Complex> b, Matrix<Complex>? c, Complex alpha, Complex beta)
    {
        var result = new Matrix<Complex>(a.Rows, b.Columns);

        for (int j = 0; j < b.Columns; j++)
        {
            for (int i = 0; i < a.Rows; i++)
            {
                Complex sum = Complex.Zero;
                for (int l = 0; l < a.Columns; l++) sum += a[i, l] * b[l, j];

                Complex prior = c is null || beta == Complex.Zero ? Complex.Zero : beta * c[i, j];
                result[i, j] = prior + (alpha * sum);
            }
        }

        return result;
    }

    /// <summary>
    /// |actual - expected|_ij &lt;= 4*k*u*|alpha|*(|A||B|)_ij, plus a
    /// few ulps of the result for the beta term and the final combine.
    /// </summary>
    internal static void AssertWithinBound(
        Matrix<Complex> actual, Matrix<Complex> a, Matrix<Complex> b, Matrix<Complex> expected,
        Complex? alpha = null, Complex? beta = null)
    {
        const double U = 1.1102230246251565e-16;

        double alphaMagnitude = Complex.Abs(alpha ?? Complex.One);
        int k = Math.Max(a.Columns, 1);

        for (int j = 0; j < actual.Columns; j++)
        {
            for (int i = 0; i < actual.Rows; i++)
            {
                double magnitudes = 0.0;
                for (int l = 0; l < a.Columns; l++) magnitudes += Complex.Abs(a[i, l]) * Complex.Abs(b[l, j]);

                double bound = (4.0 * k * U * alphaMagnitude * magnitudes) + (8.0 * U * Complex.Abs(expected[i, j])) + 1e-300;
                double error = Complex.Abs(actual[i, j] - expected[i, j]);

                Assert.True(
                    error <= bound,
                    $"[{i},{j}] error {error:E3} exceeds bound {bound:E3} (alpha {alpha}, beta {beta})");
            }
        }
    }

    internal static void Fill(Matrix<Complex> m, int seed)
    {
        var rng = new Random(seed);

        for (int j = 0; j < m.Columns; j++)
            for (int i = 0; i < m.Rows; i++)
                m[i, j] = new Complex((rng.NextDouble() * 2.0) - 1.0, (rng.NextDouble() * 2.0) - 1.0);
    }
}

public sealed class ScalarComplexMultiplyTests : ComplexMultiplyContract<ScalarCase>;
public sealed class Avx2ComplexMultiplyTests : ComplexMultiplyContract<Avx2Case>;
public sealed class Avx512ComplexMultiplyTests : ComplexMultiplyContract<Avx512Case>;

/// <summary>
/// The parts of the complex product that are not per-kernel: the 3M
/// distinction, degenerate shapes, failure behaviour, and the element-kernel
/// contract that <see cref="DoubleKernels"/> and <see cref="ComplexKernels"/>
/// must both keep.
/// </summary>
public class ComplexMultiplyTests
{
    private const double U = 1.1102230246251565e-16;

    // ---- why 4M and not 3M ----------------------------------------------------

    /// <summary>
    /// Operands whose imaginary parts are a billion times smaller than their
    /// real parts, so the product's imaginary part is too. 4M computes that
    /// part as Ar*Bi + Ai*Br -- only small terms -- and must get it to relative
    /// accuracy near k*u. 3M computes it as (Ar+Ai)(Br+Bi) - Ar*Br - Ai*Bi, a
    /// difference of large terms, and cannot.
    /// </summary>
    [Fact]
    public void ASmallImaginaryComponentKeepsItsRelativeAccuracy()
    {
        var (a, b) = NearlyRealOperands(out double epsilon);

        var product = a.Multiply(b.ReadOnlyView);

        AssertImaginaryWithinSmallComponentBound(product, a, b, epsilon, "4M");
    }

    /// <summary>
    /// The previous test has teeth: the same bound, applied to 3M computed
    /// from the same real GEMM, fails. Without this, a test built "to fail
    /// under 3M" would be a claim rather than a demonstration.
    /// </summary>
    [Fact]
    public void TheSameBoundRejectsThe3MMethod()
    {
        var (a, b) = NearlyRealOperands(out double epsilon);

        var product = ThreeM(a, b);

        var failure = Assert.ThrowsAny<Exception>(
            () => AssertImaginaryWithinSmallComponentBound(product, a, b, epsilon, "3M"));

        Assert.Contains("3M", failure.Message, StringComparison.Ordinal);
    }

    // ---- degenerate shapes and failure behaviour ------------------------------

    [Fact]
    public void AnEmptyInnerDimensionScalesByBeta()
    {
        var a = new Matrix<Complex>(3, 0);
        var b = new Matrix<Complex>(0, 4);
        var c = new Matrix<Complex>(3, 4);
        ComplexMultiplyContract<ScalarCase>.Fill(c, seed: 9);

        var before = c.Clone();
        var beta = new Complex(0.5, -1.0);

        Workspace.Shared.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View, Complex.One, beta);

        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 3; i++)
                Assert.Equal(beta * before[i, j], c[i, j]);
    }

    [Fact]
    public void AnEmptyDestinationIsANoOp()
    {
        var a = new Matrix<Complex>(0, 5);
        var b = new Matrix<Complex>(5, 1_000_000);
        var c = new Matrix<Complex>(0, 1_000_000);

        Workspace.Shared.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View);
    }

    /// <summary>beta = 0 overwrites rather than scales, so NaN in the destination cannot survive.</summary>
    [Fact]
    public void AZeroBetaOverwritesNaN()
    {
        var a = new Matrix<Complex>(4, 3);
        var b = new Matrix<Complex>(3, 5);
        var c = new Matrix<Complex>(4, 5);

        ComplexMultiplyContract<ScalarCase>.Fill(a, seed: 10);
        ComplexMultiplyContract<ScalarCase>.Fill(b, seed: 11);

        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 4; i++)
                c[i, j] = new Complex(double.NaN, double.NaN);

        Workspace.Shared.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View, new Complex(2.0, 1.0), Complex.Zero);

        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 4; i++)
                Assert.True(double.IsFinite(c[i, j].Real) && double.IsFinite(c[i, j].Imaginary), $"[{i},{j}] = {c[i, j]}");
    }

    [Fact]
    public void NonConformableShapesAreRejected()
    {
        var a = new Matrix<Complex>(3, 4);
        var b = new Matrix<Complex>(5, 2);
        var c = new Matrix<Complex>(3, 2);
        var wrong = new Matrix<Complex>(2, 2);

        Assert.Contains("Inner dimensions", Assert.Throws<ArgumentException>(
            () => Workspace.Shared.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View)).Message, StringComparison.Ordinal);

        var b2 = new Matrix<Complex>(4, 2);
        Assert.Contains("Destination", Assert.Throws<ArgumentException>(
            () => Workspace.Shared.Multiply(a.ReadOnlyView, b2.ReadOnlyView, wrong.View)).Message, StringComparison.Ordinal);
    }

    /// <summary>
    /// The destination is written only after all four real products succeed,
    /// so a failure part-way leaves it exactly as it was. A disposed workspace
    /// fails before any buffer is handed out.
    /// </summary>
    [Fact]
    public void AFailedProductLeavesTheDestinationUntouched()
    {
        var workspace = new Workspace(multithreaded: false);
        workspace.Dispose();

        var a = new Matrix<Complex>(6, 5);
        var b = new Matrix<Complex>(5, 4);
        var c = new Matrix<Complex>(6, 4);

        ComplexMultiplyContract<ScalarCase>.Fill(a, seed: 12);
        ComplexMultiplyContract<ScalarCase>.Fill(b, seed: 13);
        ComplexMultiplyContract<ScalarCase>.Fill(c, seed: 14);

        var before = c.Clone();

        Assert.Throws<ObjectDisposedException>(
            () => workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View, new Complex(1, 1), Complex.One));

        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 6; i++)
                Assert.Equal(before[i, j], c[i, j]);
    }

    /// <summary>
    /// Real operands passed as complex: the imaginary products are all of
    /// zeros, so the imaginary part is exactly zero and the real part is the
    /// real GEMM's result on the same data.
    /// </summary>
    [Fact]
    public void RealOperandsGiveTheRealProduct()
    {
        var realA = new Matrix<double>(33, 21);
        var realB = new Matrix<double>(21, 17);
        var rng = new Random(15);

        for (int j = 0; j < 21; j++) for (int i = 0; i < 33; i++) realA[i, j] = rng.NextDouble() - 0.5;
        for (int j = 0; j < 17; j++) for (int i = 0; i < 21; i++) realB[i, j] = rng.NextDouble() - 0.5;

        var complexA = new Matrix<Complex>(33, 21);
        var complexB = new Matrix<Complex>(21, 17);
        for (int j = 0; j < 21; j++) for (int i = 0; i < 33; i++) complexA[i, j] = realA[i, j];
        for (int j = 0; j < 17; j++) for (int i = 0; i < 21; i++) complexB[i, j] = realB[i, j];

        var expected = realA.Multiply(realB.ReadOnlyView);
        var actual = complexA.Multiply(complexB.ReadOnlyView);

        for (int j = 0; j < 17; j++)
        {
            for (int i = 0; i < 33; i++)
            {
                Assert.Equal(0.0, actual[i, j].Imaginary);
                Assert.Equal(expected[i, j], actual[i, j].Real, 15);
            }
        }
    }

    [Fact]
    public void TheFluentFormsAgreeWithTheWorkspace()
    {
        var a = new Matrix<Complex>(9, 7);
        var b = new Matrix<Complex>(7, 8);
        ComplexMultiplyContract<ScalarCase>.Fill(a, seed: 16);
        ComplexMultiplyContract<ScalarCase>.Fill(b, seed: 17);

        var viaWorkspace = new Matrix<Complex>(9, 8);
        Workspace.Shared.Multiply(a.ReadOnlyView, b.ReadOnlyView, viaWorkspace.View);

        var viaMultiply = a.Multiply(b);

        var viaInto = new Matrix<Complex>(9, 8);
        a.MultiplyInto(b, viaInto.View);

        var accumulated = viaWorkspace.Clone();
        a.MultiplyInto(b, accumulated.View, new Complex(-1.0, 0.0), Complex.One);

        for (int j = 0; j < 8; j++)
        {
            for (int i = 0; i < 9; i++)
            {
                Assert.Equal(viaWorkspace[i, j], viaMultiply[i, j]);
                Assert.Equal(viaWorkspace[i, j], viaInto[i, j]);
                Assert.True(Complex.Abs(accumulated[i, j]) <= 1e-14, $"C - AB = {accumulated[i, j]}");
            }
        }
    }

    // ---- the element-kernel contract, for both element types -------------------

    /// <summary>
    /// The one contract every <see cref="IElementKernels{T}"/> implementation
    /// must keep, written once over T and run for both. This is what it means
    /// for the interface to be shaped by two implementations rather than one:
    /// a member that could only be implemented for one type would fail to
    /// compile here, and one implemented inconsistently would fail to pass.
    /// </summary>
    [Fact]
    public void DoubleKernelsKeepTheContract() =>
        CheckElementContract<double, DoubleKernels>(rng => rng.NextDouble() - 0.5, double.NaN);

    /// <inheritdoc cref="DoubleKernelsKeepTheContract"/>
    [Fact]
    public void ComplexKernelsKeepTheContract() =>
        CheckElementContract<Complex, ComplexKernels>(
            rng => new Complex(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5), new Complex(double.NaN, double.NaN));

    private static void CheckElementContract<T, TKernels>(Func<Random, T> sample, T nan)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        const int M = 13, N = 11, K = 9;
        var rng = new Random(18);

        var a = new Matrix<T>(M, K);
        var b = new Matrix<T>(K, N);
        var c = new Matrix<T>(M, N);

        for (int j = 0; j < K; j++) for (int i = 0; i < M; i++) a[i, j] = sample(rng);
        for (int j = 0; j < N; j++) for (int i = 0; i < K; i++) b[i, j] = sample(rng);
        for (int j = 0; j < N; j++) for (int i = 0; i < M; i++) c[i, j] = sample(rng);

        T alpha = sample(rng);
        T beta = sample(rng);

        // Multiply, against the textbook loop in T's own arithmetic.
        var expected = new Matrix<T>(M, N);
        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < M; i++)
            {
                T sum = T.Zero;
                for (int l = 0; l < K; l++) sum += a[i, l] * b[l, j];
                expected[i, j] = (beta * c[i, j]) + (alpha * sum);
            }
        }

        TKernels.Multiply(Workspace.Shared, a.ReadOnlyView, b.ReadOnlyView, c.View, alpha, beta);

        for (int j = 0; j < N; j++)
        {
            for (int i = 0; i < M; i++)
            {
                double magnitudes = TKernels.Magnitude(beta) * TKernels.Magnitude(c[i, j]);
                for (int l = 0; l < K; l++)
                    magnitudes += TKernels.Magnitude(alpha) * TKernels.Magnitude(a[i, l]) * TKernels.Magnitude(b[l, j]);

                double error = TKernels.Magnitude(c[i, j] - expected[i, j]);
                Assert.True(error <= (8.0 * K * U * magnitudes) + 1e-300, $"{typeof(T).Name} [{i},{j}]: {error:E3}");
            }
        }

        // A zero beta overwrites: NaN in the destination cannot survive.
        var poisoned = new Matrix<T>(M, N);
        for (int j = 0; j < N; j++) for (int i = 0; i < M; i++) poisoned[i, j] = nan;

        TKernels.Multiply(Workspace.Shared, a.ReadOnlyView, b.ReadOnlyView, poisoned.View, T.One, T.Zero);

        for (int j = 0; j < N; j++)
            for (int i = 0; i < M; i++)
                Assert.False(T.IsNaN(poisoned[i, j]), $"{typeof(T).Name} [{i},{j}] kept a NaN through beta = 0");

        // Magnitude is a real modulus, zero at zero.
        Assert.Equal(0.0, TKernels.Magnitude(T.Zero));
        Assert.Equal(1.0, TKernels.Magnitude(T.One));

        // The exact norms, against their definitions written with Magnitude.
        double oneNorm = 0.0;
        for (int j = 0; j < K; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < M; i++) sum += TKernels.Magnitude(a[i, j]);
            oneNorm = Math.Max(oneNorm, sum);
        }

        double infinityNorm = 0.0;
        for (int i = 0; i < M; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < K; j++) sum += TKernels.Magnitude(a[i, j]);
            infinityNorm = Math.Max(infinityNorm, sum);
        }

        Assert.Equal(oneNorm, TKernels.OneNorm(a.ReadOnlyView), 12);
        Assert.Equal(infinityNorm, TKernels.InfinityNorm(a.ReadOnlyView), 12);

        // An empty operand has zero norm, however many columns it nominally has.
        Assert.Equal(0.0, TKernels.OneNorm(new Matrix<T>(0, 1000).ReadOnlyView));
        Assert.Equal(0.0, TKernels.InfinityNorm(new Matrix<T>(1000, 0).ReadOnlyView));
    }

    // ---- helpers ---------------------------------------------------------------

    /// <summary>A = X + i*eps*Y and B = U + i*eps*V, with eps = 1e-9.</summary>
    private static (Matrix<Complex> A, Matrix<Complex> B) NearlyRealOperands(out double epsilon)
    {
        const int M = 24, K = 40, N = 20;
        epsilon = 1e-9;

        var rng = new Random(19);
        var a = new Matrix<Complex>(M, K);
        var b = new Matrix<Complex>(K, N);

        for (int j = 0; j < K; j++)
            for (int i = 0; i < M; i++)
                a[i, j] = new Complex(rng.NextDouble() - 0.5, epsilon * (rng.NextDouble() - 0.5));

        for (int j = 0; j < N; j++)
            for (int i = 0; i < K; i++)
                b[i, j] = new Complex(rng.NextDouble() - 0.5, epsilon * (rng.NextDouble() - 0.5));

        return (a, b);
    }

    /// <summary>
    /// The imaginary part against a reference accumulated term by term from
    /// only the small products, with a bound proportional to those products
    /// alone: |Im error|_ij &lt;= 4*k*u * sum_l (|Re a||Im b| + |Im a||Re b|).
    /// </summary>
    private static void AssertImaginaryWithinSmallComponentBound(
        Matrix<Complex> product, Matrix<Complex> a, Matrix<Complex> b, double epsilon, string method)
    {
        int k = a.Columns;

        for (int j = 0; j < product.Columns; j++)
        {
            for (int i = 0; i < product.Rows; i++)
            {
                double reference = 0.0;
                double smallTerms = 0.0;

                for (int l = 0; l < k; l++)
                {
                    reference += (a[i, l].Real * b[l, j].Imaginary) + (a[i, l].Imaginary * b[l, j].Real);
                    smallTerms += (Math.Abs(a[i, l].Real) * Math.Abs(b[l, j].Imaginary))
                        + (Math.Abs(a[i, l].Imaginary) * Math.Abs(b[l, j].Real));
                }

                double bound = 4.0 * k * U * smallTerms;
                double error = Math.Abs(product[i, j].Imaginary - reference);

                Assert.True(
                    error <= bound,
                    $"{method}: [{i},{j}] imaginary error {error:E3} exceeds the small-component bound {bound:E3} (eps {epsilon:E1})");
            }
        }
    }

    /// <summary>
    /// The 3M method, built here from the library's real GEMM purely so the
    /// suite can show it fails the bound 4M passes. It is not in the library.
    /// </summary>
    private static Matrix<Complex> ThreeM(Matrix<Complex> a, Matrix<Complex> b)
    {
        int m = a.Rows, k = a.Columns, n = b.Columns;

        var ar = new Matrix<double>(m, k);
        var ai = new Matrix<double>(m, k);
        var asum = new Matrix<double>(m, k);
        var br = new Matrix<double>(k, n);
        var bi = new Matrix<double>(k, n);
        var bsum = new Matrix<double>(k, n);

        for (int j = 0; j < k; j++)
            for (int i = 0; i < m; i++)
            {
                ar[i, j] = a[i, j].Real;
                ai[i, j] = a[i, j].Imaginary;
                asum[i, j] = a[i, j].Real + a[i, j].Imaginary;
            }

        for (int j = 0; j < n; j++)
            for (int i = 0; i < k; i++)
            {
                br[i, j] = b[i, j].Real;
                bi[i, j] = b[i, j].Imaginary;
                bsum[i, j] = b[i, j].Real + b[i, j].Imaginary;
            }

        var t1 = ar.Multiply(br.ReadOnlyView);
        var t2 = ai.Multiply(bi.ReadOnlyView);
        var t3 = asum.Multiply(bsum.ReadOnlyView);

        var result = new Matrix<Complex>(m, n);
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                result[i, j] = new Complex(t1[i, j] - t2[i, j], t3[i, j] - t1[i, j] - t2[i, j]);

        return result;
    }
}
