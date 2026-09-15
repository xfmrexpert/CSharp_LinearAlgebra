using System.Diagnostics;
using System.Runtime.InteropServices;

namespace GemmLab;

/// <summary>
/// Times the micro-kernel in isolation, on packed panels small enough to stay
/// resident in L1/L2, with a single C tile held in registers across the whole
/// k-loop.
///
/// This is the kernel's own ceiling: no packing cost, no cache blocking, no
/// memory traffic beyond streaming the two panels. Comparing full GEMM against
/// this number separates "the kernel is slow" from "everything around the
/// kernel is slow" — which are completely different problems with completely
/// different fixes.
/// </summary>
public static unsafe class KernelProbe
{
    public static double MeasureGflops<TKernel>() where TKernel : struct, IMicroKernel =>
        Measure<TKernel>().Gflops;

    public static (double Gflops, TimingStatistics Timing) Measure<TKernel>() where TKernel : struct, IMicroKernel
    {
        int mr = TKernel.Mr;
        int nr = TKernel.Nr;
        const int Kc = 256;           // panels: mr*Kc + nr*Kc doubles, L1/L2 resident
        const int Reps = 200_000;

        double* ap = Alloc((nuint)(mr * Kc));
        double* bp = Alloc((nuint)(nr * Kc));
        double* c = Alloc((nuint)(mr * nr));

        var rng = new Random(7);
        for (int i = 0; i < mr * Kc; i++) ap[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < nr * Kc; i++) bp[i] = rng.NextDouble() - 0.5;
        new Span<double>(c, mr * nr).Clear();

        // Warm up.
        for (int r = 0; r < 1000; r++)
            TKernel.Execute(Kc, 1.0, ap, bp, c, mr);

        var samples = new double[TimingStatistics.SampleCount];
        for (int trial = 0; trial < samples.Length; trial++)
        {
            new Span<double>(c, mr * nr).Clear();
            long started = Stopwatch.GetTimestamp();
            for (int r = 0; r < Reps; r++)
                TKernel.Execute(Kc, 1.0, ap, bp, c, mr);
            samples[trial] = Stopwatch.GetElapsedTime(started).TotalSeconds;
        }

        NativeMemory.AlignedFree(ap);
        NativeMemory.AlignedFree(bp);
        NativeMemory.AlignedFree(c);

        TimingStatistics timing = TimingStatistics.FromSamples(samples);
        double flops = 2.0 * mr * nr * Kc * Reps;
        return (flops / timing.MedianSeconds / 1e9, timing);
    }

    private static double* Alloc(nuint count) =>
        (double*)NativeMemory.AlignedAlloc(count * sizeof(double), 64);
}
