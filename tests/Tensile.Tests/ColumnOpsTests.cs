using Tensile.Kernels;

namespace Tensile.Tests;

/// <summary>
/// The streamed column updates, checked against the obvious scalar loop.
///
/// These are hand-vectorised, so their tail handling and their lane reductions
/// are code the triangular solves and LU's panel exercise only incidentally.
/// </summary>
public unsafe class ColumnOpsTests
{
    /// <summary>Lengths straddling one, two and four vector widths, plus tails.</summary>
    public static TheoryData<int> Lengths => new() { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 1000 };

    [Theory]
    [MemberData(nameof(Lengths))]
    public void DotMatchesScalarLoop(int n)
    {
        using var x = TestMatrix.Random(Math.Max(n, 1), 1, seed: 200 + n);
        using var y = TestMatrix.Random(Math.Max(n, 1), 1, seed: 300 + n);

        double expected = 0.0;
        for (int i = 0; i < n; i++) expected += x[i, 0] * y[i, 0];

        double actual = ColumnOps.Dot(n, x.Data, y.Data);

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

        ColumnOps.Scale(n, 2.75, x.Data);

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

        ColumnOps.Axpy(n, -1.25, x.Data, y.Data);

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

        ColumnOps.Scale(n, 3.0, x.Data);
        ColumnOps.Axpy(n, 3.0, x.Data, y.Data);

        for (int i = n; i < y.Rows; i++)
        {
            Assert.Equal(Sentinel, x[i, 0]);
            Assert.Equal(Sentinel, y[i, 0]);
        }
    }
}
