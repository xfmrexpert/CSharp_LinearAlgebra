using Tensile.Kernels;

namespace Tensile.Tests;

/// <summary>
/// Pivot selection, checked against the obvious scalar scan.
///
/// Worth testing directly and not only through LU: the scan is hand-vectorised,
/// so its tail handling and its lane reduction are code LU exercises only
/// incidentally, and a wrong tie rule in <see cref="Pivoting.IndexOfMaxAbs"/>
/// would change the pivot sequence without making any residual worse.
/// </summary>
public unsafe class PivotingTests
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

        Assert.Equal(expected, Pivoting.IndexOfMaxAbs(n, x.Data));
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

        Assert.Equal(0, Pivoting.IndexOfMaxAbs(n, x.Data));
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

            Assert.Equal(position, Pivoting.IndexOfMaxAbs(n, x.Data));
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

        Assert.Equal(23, Pivoting.IndexOfMaxAbs(n, x.Data));
    }

    [Fact]
    public void IndexOfMaxAbsReturnsZeroForEmpty() => Assert.Equal(0, Pivoting.IndexOfMaxAbs(0, null));
}
