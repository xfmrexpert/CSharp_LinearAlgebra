using Tensile.Kernels;

namespace Tensile.Tests;

/// <summary>
/// The level-1 primitives, checked against the obvious scalar loop.
///
/// These are worth testing directly and not only through LU: they are
/// hand-vectorised, so their tail handling and their lane reductions are code
/// that LU exercises only incidentally, and a wrong tie rule in
/// <see cref="Blas1.IndexOfMaxAbs"/> would change the pivot sequence without
/// making any residual worse.
/// </summary>
public unsafe class Blas1Tests
{
    /// <summary>Lengths straddling one, two and four vector widths, plus tails.</summary>
    public static TheoryData<int> Lengths => new() { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 1000 };

    [Theory]
    [MemberData(nameof(Lengths))]
    public void IndexOfMaxAbsMatchesScalarScan(int n)
    {
        if (n == 0) return;

        using var x = TestMatrix.Random(n, 1, seed: 100 + n);

        int expected = 0;
        double best = Math.Abs(x[0, 0]);

        for (int i = 1; i < n; i++)
        {
            double value = Math.Abs(x[i, 0]);
            if (value > best) { best = value; expected = i; }
        }

        Assert.Equal(expected, Blas1.IndexOfMaxAbs(n, x.Data));
    }

    /// <summary>
    /// With every entry equal, the lowest index must win. The vectorised scan
    /// carries a per-lane candidate, so a reduction that took the last lane
    /// rather than the lowest index would show up here and nowhere else.
    /// </summary>
    [Theory]
    [MemberData(nameof(Lengths))]
    public void IndexOfMaxAbsBreaksTiesToTheLowestIndex(int n)
    {
        if (n == 0) return;

        using var x = new TestMatrix(n, 1);
        for (int i = 0; i < n; i++) x[i, 0] = i % 2 == 0 ? 3.0 : -3.0;

        Assert.Equal(0, Blas1.IndexOfMaxAbs(n, x.Data));
    }

    /// <summary>A maximum placed in each position in turn must be found there.</summary>
    [Fact]
    public void IndexOfMaxAbsFindsEveryPosition()
    {
        const int n = 37;

        for (int position = 0; position < n; position++)
        {
            using var x = new TestMatrix(n, 1);
            for (int i = 0; i < n; i++) x[i, 0] = 1.0;
            x[position, 0] = -9.0;

            Assert.Equal(position, Blas1.IndexOfMaxAbs(n, x.Data));
        }
    }

    /// <summary>
    /// NaN never compares greater, so it must never be selected -- matching the
    /// scalar loop, and matching idamax.
    /// </summary>
    [Fact]
    public void IndexOfMaxAbsIgnoresNaN()
    {
        const int n = 40;

        using var x = new TestMatrix(n, 1);
        for (int i = 0; i < n; i++) x[i, 0] = 1.0;
        x[7, 0] = double.NaN;
        x[23, 0] = 5.0;

        Assert.Equal(23, Blas1.IndexOfMaxAbs(n, x.Data));
    }

    [Fact]
    public void IndexOfMaxAbsReturnsZeroForEmpty() => Assert.Equal(0, Blas1.IndexOfMaxAbs(0, null));

    [Theory]
    [MemberData(nameof(Lengths))]
    public void DotMatchesScalarLoop(int n)
    {
        using var x = TestMatrix.Random(Math.Max(n, 1), 1, seed: 200 + n);
        using var y = TestMatrix.Random(Math.Max(n, 1), 1, seed: 300 + n);

        double expected = 0.0;
        for (int i = 0; i < n; i++) expected += x[i, 0] * y[i, 0];

        double actual = Blas1.Dot(n, x.Data, y.Data);

        // Four accumulators sum in a different order, so this is a residual
        // check, not an equality check.
        Assert.True(
            Math.Abs(actual - expected) <= 1e-12 * (1.0 + Math.Abs(expected)),
            $"n={n}: {actual:E17} vs {expected:E17}");
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void ScaleMatchesScalarLoop(int n)
    {
        using var x = TestMatrix.Random(Math.Max(n, 1), 1, seed: 400 + n);
        using var expected = x.Clone();

        for (int i = 0; i < n; i++) expected[i, 0] *= 2.75;

        Blas1.Scale(n, 2.75, x.Data);

        Assert.Equal(0.0, x.MaxDifference(expected));
    }

    [Theory]
    [MemberData(nameof(Lengths))]
    public void AxpyMatchesScalarLoop(int n)
    {
        using var x = TestMatrix.Random(Math.Max(n, 1), 1, seed: 500 + n);
        using var y = TestMatrix.Random(Math.Max(n, 1), 1, seed: 600 + n);
        using var expected = y.Clone();

        for (int i = 0; i < n; i++) expected[i, 0] += -1.25 * x[i, 0];

        Blas1.Axpy(n, -1.25, x.Data, y.Data);

        Assert.True(y.MaxDifference(expected) < 1e-15);
    }

    /// <summary>The vectorised paths must not run past the length they are given.</summary>
    [Theory]
    [MemberData(nameof(Lengths))]
    public void OperationsStayWithinBounds(int n)
    {
        const int Slack = 8;
        const double Sentinel = 1234.5;

        using var x = TestMatrix.Random(Math.Max(n, 1) + Slack, 1, seed: 700 + n);
        using var y = new TestMatrix(Math.Max(n, 1) + Slack, 1);

        for (int i = n; i < y.Rows; i++) { x[i, 0] = Sentinel; y[i, 0] = Sentinel; }

        Blas1.Scale(n, 3.0, x.Data);
        Blas1.Axpy(n, 3.0, x.Data, y.Data);

        for (int i = n; i < y.Rows; i++)
        {
            Assert.Equal(Sentinel, x[i, 0]);
            Assert.Equal(Sentinel, y[i, 0]);
        }
    }
}
