using System.Numerics;

namespace Tensile.Tests;

/// <summary>
/// The native complex exponentials against the embedded route they replaced.
///
/// The embedding is the oracle: the real algorithms on [[X, -Y], [Y, X]], real
/// code verified on its own, sharing none of the complex primitives the native
/// path runs on -- no 4M product, no complex LU, no complex estimator. Both
/// routes are backward stable, so they agree to a small multiple of the
/// rounding error, not bit for bit; and they may legitimately choose different
/// parameters, since the native route reads complex norms and the embedded one
/// reads real norms up to sqrt(2) larger.
///
/// Accuracy against an independent complex Taylor series, and the reductions
/// to the real path on real input, are in <see cref="ComplexExponentialTests"/>
/// and now exercise the native path; this file adds what is specific to it.
/// </summary>
public class NativeComplexExponentialTests
{
    private const double Tolerance = 1e-11;

    /// <summary>
    /// Norms chosen so the native path lands on every Padé degree and on the
    /// scaled degree-13 branch, and agrees with the oracle on each. A suite
    /// whose matrices all took one branch would leave the others unverified
    /// (CLAUDE.md, testing guidance).
    /// </summary>
    [Fact]
    public void EveryPadeDegreeIsReachedAndAgreesWithTheEmbedding()
    {
        double[] norms = [0.001, 0.05, 0.4, 1.2, 6, 40];
        var degrees = new HashSet<int>();
        bool scaled = false;

        foreach (double norm in norms)
        {
            var a = RandomWithNorm(8, norm, seed: (int)(norm * 1000) + 1);
            var diagnostics = new ExpmDiagnostics();

            var native = MatrixExponential.Expm(a, workspace: null, diagnostics);
            var embedded = ComplexEmbedding.Expm(a);

            degrees.Add(diagnostics.Degree);
            scaled |= diagnostics.Squarings > 0;

            AssertClose(native, embedded, $"norm {norm}, degree {diagnostics.Degree}");
        }

        Assert.Equal([3, 5, 7, 9, 13], degrees.Order());
        Assert.True(scaled, "no case took the scaling branch");
    }

    [Theory]
    [InlineData(5, 0.5)]
    [InlineData(20, 3.0)]
    [InlineData(40, 25.0)]
    [InlineData(64, 100.0)]
    public void ExpmAgreesWithTheEmbedding(int n, double norm)
    {
        var a = RandomWithNorm(n, norm, seed: n);

        AssertClose(a.Expm(), ComplexEmbedding.Expm(a), $"n {n}, norm {norm}");
    }

    /// <summary>
    /// A strongly nonnormal case, where the ||A^k||^(1/k) estimates matter:
    /// a complex upper triangular matrix with a large off-diagonal.
    /// </summary>
    [Fact]
    public void ANonnormalMatrixAgreesWithTheEmbedding()
    {
        const int N = 12;

        var a = new Matrix<Complex>(N, N);
        for (int i = 0; i < N; i++)
        {
            a[i, i] = new Complex(-0.5 + (0.1 * i), 0.3 * i);
            for (int j = i + 1; j < N; j++) a[i, j] = new Complex(30.0 / (j - i), -10.0 / (j - i));
        }

        AssertClose(a.Expm(), ComplexEmbedding.Expm(a), "nonnormal");
    }

    [Theory]
    [InlineData(10, 1.0, 1, 1.0)]
    [InlineData(30, 5.0, 3, 1.0)]
    [InlineData(30, 5.0, 2, -0.7)]
    [InlineData(60, 20.0, 1, 0.25)]
    public void ExpmvAgreesWithTheEmbedding(int n, double norm, int columns, double t)
    {
        var a = RandomWithNorm(n, norm, seed: n + columns);
        var b = ComplexLuTests.Random(n, columns, seed: n + 7);

        AssertClose(a.Expmv(b.ReadOnlyView, t), ComplexEmbedding.Expmv(a, b.ReadOnlyView, t), $"n {n}, t {t}");
    }

    /// <summary>
    /// A large j*omega*I on top of a small operator: the native shift removes
    /// all of it, the embedded route removes its imaginary part separately
    /// (finding 15). Both must reach the same answer; the native one must not
    /// pay for the shift at all.
    /// </summary>
    [Fact]
    public void AnImaginaryShiftCostsTheNativeRouteNothing()
    {
        const int N = 24;

        var a = RandomWithNorm(N, 2.0, seed: 11);
        var shifted = a.Clone();
        for (int i = 0; i < N; i++) shifted[i, i] += new Complex(0.0, 100.0);

        var b = ComplexLuTests.Random(N, 2, seed: 12);

        var plain = new ExpmvDiagnostics();
        var withShift = new ExpmvDiagnostics();
        _ = MatrixExponentialAction.Expmv(a, b.ReadOnlyView, 1.0, plain);
        var native = MatrixExponentialAction.Expmv(shifted, b.ReadOnlyView, 1.0, withShift);

        Assert.Equal(plain.Degree, withShift.Degree);
        Assert.Equal(plain.Scaling, withShift.Scaling);

        AssertClose(native, ComplexEmbedding.Expmv(shifted, b.ReadOnlyView, 1.0), "shifted");
    }

    [Theory]
    [InlineData(8, 1.0)]
    [InlineData(25, 6.0)]
    public void MatrixFreeAgreesWithTheEmbedding(int n, double norm)
    {
        var a = RandomWithNorm(n, norm, seed: n + 3);
        var b = ComplexLuTests.Random(n, 2, seed: n + 4);
        var op = new ApplyOnly(a);
        double bound = a.OneNorm();

        AssertClose(
            MatrixExponentialAction.Expmv(op, b.ReadOnlyView, 0.8, bound),
            ComplexEmbedding.Expmv(op, b.ReadOnlyView, 0.8, bound),
            $"matrix-free n {n}");
    }

    /// <summary>
    /// The matrix-free path uses exactly the bound it is given -- it no longer
    /// inflates it by sqrt(2) for a real representation -- so a real operator
    /// passed as complex chooses the real matrix-free parameters.
    /// </summary>
    [Fact]
    public void MatrixFreeOnARealOperatorChoosesTheRealParameters()
    {
        const int N = 20;

        var real = ComplexLuTests.RandomReal(N, N, seed: 13);
        var realB = ComplexLuTests.RandomReal(N, 1, seed: 14);
        double bound = real.OneNorm() * 3.0;

        var realDiagnostics = new ExpmvDiagnostics();
        _ = MatrixExponentialAction.Expmv(new DenseMatrixOperator(real), realB.ReadOnlyView, 1.0, bound, realDiagnostics);

        var complexDiagnostics = new ExpmvDiagnostics();
        _ = MatrixExponentialAction.Expmv(new ApplyOnly(ToComplex(real)), ToComplex(realB).ReadOnlyView, 1.0, bound, complexDiagnostics);

        Assert.Equal(realDiagnostics.Degree, complexDiagnostics.Degree);
        Assert.Equal(realDiagnostics.Scaling, complexDiagnostics.Scaling);
        Assert.Equal(realDiagnostics.Applications, complexDiagnostics.Applications);
    }

    /// <summary>
    /// A real matrix exponentiated as complex runs the real parameters with
    /// complex arithmetic whose imaginary parts are exactly zero throughout,
    /// so the result must be real exactly, not approximately.
    /// </summary>
    [Fact]
    public void ARealMatrixHasAnExactlyRealExponential()
    {
        var real = ComplexLuTests.RandomReal(16, 16, seed: 15);
        for (int j = 0; j < 16; j++) for (int i = 0; i < 16; i++) real[i, j] *= 3.0;

        var result = ToComplex(real).Expm();

        for (int j = 0; j < 16; j++)
            for (int i = 0; i < 16; i++)
                Assert.Equal(0.0, result[i, j].Imaginary);
    }

    // ---- helpers ---------------------------------------------------------------

    internal static Matrix<Complex> RandomWithNorm(int n, double norm, int seed)
    {
        var a = ComplexLuTests.Random(n, n, seed);
        double scale = norm / a.OneNorm();

        for (int j = 0; j < n; j++)
            for (int i = 0; i < n; i++)
                a[i, j] *= scale;

        return a;
    }

    internal static void AssertClose(Matrix<Complex> actual, Matrix<Complex> expected, string label)
    {
        double scale = Math.Max(ComplexLuTests.MaxAbs(expected.ReadOnlyView), 1e-300);
        double difference = ComplexLuTests.MaxDifference(actual.ReadOnlyView, expected.ReadOnlyView);

        Assert.True(difference <= Tolerance * scale, $"{label}: relative difference {difference / scale:E3}");
    }

    private static Matrix<Complex> ToComplex(Matrix<double> real)
    {
        var complex = new Matrix<Complex>(real.Rows, real.Columns);
        for (int j = 0; j < real.Columns; j++)
            for (int i = 0; i < real.Rows; i++)
                complex[i, j] = real[i, j];
        return complex;
    }

    /// <summary>A dense matrix behind the narrow interface, so the matrix-free paths cannot see its entries.</summary>
    private sealed class ApplyOnly(Matrix<Complex> a) : ILinearOperator<Complex>
    {
        public int Order => a.Rows;

        public void Apply(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y) =>
            Workspace.Shared.Multiply(a.ReadOnlyView, x, y);
    }
}

/// <summary>
/// The native complex exponential on every micro-kernel, serial and threaded:
/// its products are 4M on whichever GEMM the workspace owns, so each kernel is
/// a different implementation of its inner loop.
/// </summary>
public abstract class NativeComplexExponentialContract<TCase> where TCase : struct, IKernelCase
{
    internal static readonly KernelDriver Kernel = KernelDriver.For<TCase>();

    protected NativeComplexExponentialContract() =>
        Assert.SkipUnless(Kernel.IsSupported, $"{Kernel.Name} is not supported on this CPU");

    [Theory]
    [InlineData(9, 0.3, false)]
    [InlineData(33, 8.0, false)]
    [InlineData(33, 8.0, true)]
    public void ExpmAgreesWithTheEmbeddingOnThisKernel(int n, double norm, bool multithreaded)
    {
        using var workspace = Kernel.Workspace(multithreaded);

        var a = NativeComplexExponentialTests.RandomWithNorm(n, norm, seed: n * 3);

        NativeComplexExponentialTests.AssertClose(
            a.Expm(workspace), ComplexEmbedding.Expm(a, workspace), $"{Kernel.Name}, n {n}");
    }
}

public sealed class ScalarNativeComplexExponentialTests : NativeComplexExponentialContract<ScalarCase>;
public sealed class Avx2NativeComplexExponentialTests : NativeComplexExponentialContract<Avx2Case>;
public sealed class Avx512NativeComplexExponentialTests : NativeComplexExponentialContract<Avx512Case>;
