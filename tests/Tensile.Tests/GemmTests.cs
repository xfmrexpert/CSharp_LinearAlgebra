using Tensile.Kernels;

namespace Tensile.Tests;

/// <summary>
/// The GEMM contract, run against every micro-kernel the host supports.
///
/// Correctness is checked by residual against <see cref="Reference"/> rather
/// than element-wise equality: blocked GEMM sums the same products in a
/// different order, so the answers legitimately differ in the last bits.
/// </summary>
public abstract unsafe class GemmContract<TCase> where TCase : struct, IKernelCase
{
    /// <summary>The kernel this instantiation of the contract runs against.</summary>
    internal static readonly KernelDriver Kernel = KernelDriver.For<TCase>();

    /// <summary>
    /// Generous next to the few-ulps-times-sqrt(k) a correct blocked GEMM
    /// produces, but far below anything a real indexing bug would survive.
    /// </summary>
    private const double Tolerance = 1e-13;

    protected GemmContract() =>
        Assert.SkipUnless(Kernel.IsSupported, $"{Kernel.Name} is not supported on this CPU");

    /// <summary>
    /// Shapes chosen to straddle MR and NR in every combination: exact
    /// multiples, one short, one over, and prime-ish values that leave ragged
    /// edges in all three dimensions at once.
    /// </summary>
    public static TheoryData<int, int, int> Shapes => new()
    {
        { 1, 1, 1 },
        { 1, 1, 64 },
        { 8, 6, 4 },
        { 16, 8, 16 },
        { 17, 9, 5 },
        { 31, 33, 7 },
        { 48, 48, 48 },
        { 64, 64, 64 },
        { 65, 63, 66 },
        { 100, 37, 129 },
        { 129, 130, 2 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SerialMatchesReference(int m, int n, int k) =>
        CheckAgainstReference(m, n, k, alpha: 1.0, beta: 0.0, padding: 0, Serial);

    [Theory]
    [MemberData(nameof(Shapes))]
    public void ParallelMatchesReference(int m, int n, int k) =>
        CheckAgainstReference(m, n, k, alpha: 1.0, beta: 0.0, padding: 0, Parallel);

    [Theory]
    [InlineData(1.0, 0.0)]
    [InlineData(1.0, 1.0)]
    [InlineData(-1.0, 1.0)]
    [InlineData(0.0, 1.0)]
    [InlineData(0.0, 0.0)]
    [InlineData(2.5, -0.75)]
    public void ScalingMatchesReference(double alpha, double beta)
    {
        CheckAgainstReference(37, 29, 41, alpha, beta, padding: 0, Serial);
        CheckAgainstReference(37, 29, 41, alpha, beta, padding: 0, Parallel);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(17)]
    public void PaddedStridesMatchReference(int padding)
    {
        CheckAgainstReference(35, 27, 19, alpha: 1.0, beta: 0.5, padding, Serial);
        CheckAgainstReference(35, 27, 19, alpha: 1.0, beta: 0.5, padding, Parallel);
    }

    /// <summary>
    /// beta = 0 must overwrite C rather than scale it, so a C full of NaN still
    /// produces a finite result. Scaling by zero instead would propagate the
    /// NaN, which is the classic BLAS trap and is why ScaleC special-cases it.
    /// </summary>
    [Fact]
    public void ZeroBetaOverwritesRatherThanScales()
    {
        const int m = 33, n = 21, k = 12;

        using var a = TestMatrix.Random(m, k, seed: 1);
        using var b = TestMatrix.Random(k, n, seed: 2);
        using var c = new TestMatrix(m, n);

        c.FillAll(double.NaN);

        using var gemm = Kernel.Serial();
        Kernel.MultiplySerial(gemm, m, n, k, 1.0, a.Data, a.Stride, b.Data, b.Stride, 0.0, c.Data, c.Stride);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                Assert.True(double.IsFinite(c[i, j]), $"C[{i},{j}] is not finite");
    }

    /// <summary>
    /// Nothing outside the logical m x n region of C may be touched, including
    /// the padding rows between one column and the next.
    /// </summary>
    [Fact]
    public void PaddingIsNotWritten()
    {
        const int m = 29, n = 23, k = 31, padding = 5;

        using var a = TestMatrix.Random(m, k, seed: 3, stride: m + padding);
        using var b = TestMatrix.Random(k, n, seed: 4, stride: k + padding);
        using var c = new TestMatrix(m, n, stride: m + padding);

        const double Sentinel = -12345.5;
        c.FillAll(Sentinel);

        using var gemm = Kernel.Multithreaded();
        Kernel.Multiply(gemm, m, n, k, 1.0, a.Data, a.Stride, b.Data, b.Stride, 0.0, c.Data, c.Stride);

        for (int j = 0; j < n; j++)
            for (int i = m; i < c.Stride; i++)
                Assert.Equal(Sentinel, c[i, j]);

        // The trailing columns of the allocation are padding too.
        for (int slot = n * c.Stride; slot < c.Count; slot++)
            Assert.Equal(Sentinel, c.Data[slot]);
    }

    /// <summary>
    /// The dispatch must agree with both of the paths it chooses between,
    /// across the threshold in both directions.
    /// </summary>
    [Fact]
    public void DispatchAgreesWithBothPaths()
    {
        foreach ((int m, int n, int k) in new[] { (16, 16, 16), (192, 193, 194) })
        {
            using var a = TestMatrix.Random(m, k, seed: 5);
            using var b = TestMatrix.Random(k, n, seed: 6);
            using var viaDispatch = new TestMatrix(m, n);
            using var viaSerial = new TestMatrix(m, n);

            using var gemm = Kernel.Multithreaded();

            Kernel.Multiply(gemm, m, n, k, 1.0, a.Data, a.Stride, b.Data, b.Stride,
                0.0, viaDispatch.Data, viaDispatch.Stride);
            Kernel.MultiplySerial(gemm, m, n, k, 1.0, a.Data, a.Stride, b.Data, b.Stride,
                0.0, viaSerial.Data, viaSerial.Stride);

            double residual = Reference.RelativeResidual(
                m, n, viaDispatch.Data, viaDispatch.Stride, viaSerial.Data, viaSerial.Stride);

            Assert.True(residual < Tolerance, $"{m}x{n}x{k}: dispatch vs serial residual {residual:E3}");
        }
    }

    /// <summary>A zero-sized operand must be a no-op, not a crash.</summary>
    [Theory]
    [InlineData(0, 4, 4)]
    [InlineData(4, 0, 4)]
    [InlineData(4, 4, 0)]
    public void DegenerateShapesAreHandled(int m, int n, int k)
    {
        using var a = TestMatrix.Random(Math.Max(m, 1), Math.Max(k, 1), seed: 7);
        using var b = TestMatrix.Random(Math.Max(k, 1), Math.Max(n, 1), seed: 8);
        using var c = new TestMatrix(Math.Max(m, 1), Math.Max(n, 1));

        using var gemm = Kernel.Multithreaded();

        Kernel.Multiply(gemm, m, n, k, 1.0, a.Data, a.Stride, b.Data, b.Stride, 1.0, c.Data, c.Stride);
        Kernel.MultiplySerial(gemm, m, n, k, 1.0, a.Data, a.Stride, b.Data, b.Stride, 1.0, c.Data, c.Stride);
    }

    private enum Path { Serial, Parallel }

    private const Path Serial = Path.Serial;
    private const Path Parallel = Path.Parallel;

    private static void CheckAgainstReference(
        int m, int n, int k, double alpha, double beta, int padding, Path path)
    {
        using var a = TestMatrix.Random(m, k, seed: 11, stride: m + padding);
        using var b = TestMatrix.Random(k, n, seed: 12, stride: k + padding);
        using var c = TestMatrix.Random(m, n, seed: 13, stride: m + padding);
        using var expected = c.Clone();

        Reference.Multiply(m, n, k, alpha, a.Data, a.Stride, b.Data, b.Stride,
            beta, expected.Data, expected.Stride);

        using var gemm = Kernel.Multithreaded();

        if (path == Path.Serial)
        {
            Kernel.MultiplySerial(gemm, m, n, k, alpha, a.Data, a.Stride, b.Data, b.Stride,
                beta, c.Data, c.Stride);
        }
        else
        {
            Kernel.MultiplyParallel(m, n, k, alpha, a.Data, a.Stride, b.Data, b.Stride,
                beta, c.Data, c.Stride);
        }

        double residual = Reference.RelativeResidual(
            m, n, c.Data, c.Stride, expected.Data, expected.Stride);

        Assert.True(
            residual < Tolerance,
            $"{path} {m}x{n}x{k} alpha={alpha} beta={beta} padding={padding}: residual {residual:E3}");
    }
}

public sealed class ScalarGemmTests : GemmContract<ScalarCase>;
public sealed class Avx2GemmTests : GemmContract<Avx2Case>;
public sealed class Avx512GemmTests : GemmContract<Avx512Case>;
