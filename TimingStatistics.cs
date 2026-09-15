namespace GemmLab;

public readonly record struct TimingStatistics(
    double MedianSeconds, double FirstQuartileSeconds, double ThirdQuartileSeconds)
{
    public const int SampleCount = 31;

    public double IqrSeconds => ThirdQuartileSeconds - FirstQuartileSeconds;

    public static TimingStatistics FromSamples(double[] samples)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (samples.Length == 0 || samples.Any(sample => !double.IsFinite(sample) || sample <= 0))
            throw new ArgumentException("Timing samples must be finite, positive, and nonempty.", nameof(samples));

        Array.Sort(samples);
        return new TimingStatistics(
            Quantile(samples, 0.5), Quantile(samples, 0.25), Quantile(samples, 0.75));
    }

    private static double Quantile(double[] sortedSamples, double probability)
    {
        double position = (sortedSamples.Length - 1) * probability;
        int lower = (int)position;
        int upper = Math.Min(lower + 1, sortedSamples.Length - 1);
        return sortedSamples[lower] + (position - lower) * (sortedSamples[upper] - sortedSamples[lower]);
    }
}