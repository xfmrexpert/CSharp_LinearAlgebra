namespace GemmLab.Tests;

/// <summary>
/// The timing statistics, because every performance claim in this repository is
/// reported through them. An interpolation bug here would not crash anything,
/// it would just quietly move every number.
/// </summary>
public class TimingStatisticsTests
{
    /// <summary>
    /// 1..9 has median 5, and linearly interpolated quartiles at positions
    /// 8*0.25 = 2 and 8*0.75 = 6, so exactly 3 and 7 with no interpolation.
    /// </summary>
    [Fact]
    public void OddCountUsesExactPositions()
    {
        var stats = TimingStatistics.FromSamples([1, 2, 3, 4, 5, 6, 7, 8, 9]);

        Assert.Equal(5.0, stats.MedianSeconds, 12);
        Assert.Equal(3.0, stats.FirstQuartileSeconds, 12);
        Assert.Equal(7.0, stats.ThirdQuartileSeconds, 12);
        Assert.Equal(4.0, stats.IqrSeconds, 12);
    }

    /// <summary>
    /// 1..4: median sits at position 1.5, so halfway between 2 and 3. The
    /// quartiles land at 0.75 and 2.25 and are interpolated.
    /// </summary>
    [Fact]
    public void EvenCountInterpolates()
    {
        var stats = TimingStatistics.FromSamples([1, 2, 3, 4]);

        Assert.Equal(2.5, stats.MedianSeconds, 12);
        Assert.Equal(1.75, stats.FirstQuartileSeconds, 12);
        Assert.Equal(3.25, stats.ThirdQuartileSeconds, 12);
    }

    [Fact]
    public void OrderOfSamplesDoesNotMatter()
    {
        var ordered = TimingStatistics.FromSamples([1, 2, 3, 4, 5, 6, 7]);
        var shuffled = TimingStatistics.FromSamples([5, 1, 7, 3, 6, 2, 4]);

        Assert.Equal(ordered, shuffled);
    }

    [Fact]
    public void SingleSampleIsItsOwnEveryQuantile()
    {
        var stats = TimingStatistics.FromSamples([2.5]);

        Assert.Equal(2.5, stats.MedianSeconds, 12);
        Assert.Equal(0.0, stats.IqrSeconds, 12);
    }

    /// <summary>
    /// A non-positive or non-finite sample means the measurement is broken, and
    /// silently averaging it in would corrupt a whole benchmark table.
    /// </summary>
    [Theory]
    [InlineData(0.0)]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void InvalidSamplesAreRejected(double bad) =>
        Assert.Throws<ArgumentException>(() => TimingStatistics.FromSamples([1.0, bad, 3.0]));

    [Fact]
    public void EmptyIsRejected() =>
        Assert.Throws<ArgumentException>(() => TimingStatistics.FromSamples([]));

    [Fact]
    public void NullIsRejected() =>
        Assert.Throws<ArgumentNullException>(() => TimingStatistics.FromSamples(null!));
}
