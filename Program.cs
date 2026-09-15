using System.Diagnostics;
using System.Numerics;
using System.Runtime;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace GemmLab;

public static unsafe class Program
{
    private static readonly int[] BenchSizes = { 128, 256, 512, 1024, 2048 };
    private static Dictionary<int, double>? _blisGflops;

    private delegate void MultiplyOperation(
        int rows, int columns, int depth,
        double alpha, double* left, int leftStride,
        double* right, int rightStride,
        double beta, double* result, int resultStride);

    public static void Main(string[] args)
    {
        bool verifyOnly = args.Contains("--verify-only");
        PrintEnvironment();
        if (!VerifyTimingStatistics()) return;

        RunBlis(verifyOnly);

        RunIfSupported<Avx512Kernel16x8>(verifyOnly);
        RunIfSupported<Avx2Kernel8x6>(verifyOnly);
        RunIfSupported<ScalarKernel4x4>(verifyOnly);

        RunParallel<Avx512Kernel16x8>(verifyOnly);
        RunParallel<Avx2Kernel8x6>(verifyOnly);

        if (!verifyOnly) BenchmarkReference(256);
    }

    private static void RunBlis(bool verifyOnly)
    {
        Console.WriteLine();
        Console.WriteLine("=== BLIS (native, single-threaded) ===");

        using var blis = Blis.TryLoad(out string reason);
        if (blis is null)
        {
            Console.WriteLine($"  unavailable, skipping: {reason}");
            return;
        }

        Console.WriteLine($"  library       : {blis.LibraryName} (BLIS {blis.Version}, {blis.IntegerBits}-bit integers)");
        Console.WriteLine($"  architecture  : {blis.Architecture}");
        Console.WriteLine($"  DGEMM kernel  : {blis.GemmKernelImplementation} (native packed micro-kernel)");
        Console.WriteLine($"  threads       : {blis.Threads}");
        if (blis.Architecture is "generic" or "unknown" || blis.GemmKernelImplementation != "optimized")
            Console.WriteLine("  WARNING: optimized BLIS dispatch is not confirmed; rebuild BLIS with ./configure auto before drawing performance conclusions.");
        if (!Verify(blis.Multiply, 16, 8) || verifyOnly) return;

        _blisGflops = Benchmark(blis.Multiply);
    }

    private static void RunIfSupported<TKernel>(bool verifyOnly) where TKernel : struct, IMicroKernel
    {
        Console.WriteLine();
        Console.WriteLine($"=== {TKernel.Name} (MR={TKernel.Mr}, NR={TKernel.Nr}) ===");

        if (!TKernel.IsSupported)
        {
            Console.WriteLine("  not supported on this CPU, skipping");
            return;
        }

        if (!Verify<TKernel>() || verifyOnly) return;

        var probe = KernelProbe.Measure<TKernel>();
        Console.WriteLine($"  kernel ceiling: {probe.Gflops:F2} GFLOP/s (median, L1-resident panels, no packing)");
        Console.WriteLine($"  probe batches : median {probe.Timing.MedianSeconds * 1e3:F3} ms, IQR {probe.Timing.IqrSeconds * 1e3:F3} ms ({TimingStatistics.SampleCount} samples)");

        Benchmark<TKernel>(probe.Gflops);
    }

    private static void PrintEnvironment()
    {
        Console.WriteLine($".NET            : {Environment.Version}");
        Console.WriteLine($"OS              : {RuntimeInformation.OSDescription}");
        Console.WriteLine($"Arch            : {RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"Logical cores   : {Environment.ProcessorCount}");
        Console.WriteLine($"Vector<double>  : {Vector<double>.Count} lanes");
        Console.WriteLine($"AVX2 / FMA      : {Avx2.IsSupported} / {Fma.IsSupported}");
        Console.WriteLine($"AVX-512F        : {Avx512F.IsSupported}");
        Console.WriteLine($"Server GC       : {GCSettings.IsServerGC}");
        PrintAffinity();
        Console.WriteLine($"Timing          : median and IQR (Q3 - Q1), {TimingStatistics.SampleCount} samples per size/probe");
    }

    private static void PrintAffinity()
    {
        if (OperatingSystem.IsLinux() || OperatingSystem.IsWindows())
        {
            try
            {
                using var process = Process.GetCurrentProcess();
                ulong mask = (ulong)(nuint)process.ProcessorAffinity;
                if (BitOperations.PopCount(mask) == 1)
                {
                    Console.WriteLine($"CPU affinity    : CPU {BitOperations.TrailingZeroCount(mask)} (mask 0x{mask:X}, pinned)");
                    return;
                }

                Console.WriteLine($"CPU affinity    : mask 0x{mask:X} (not pinned to one logical CPU)");
            }
            catch (Exception error) when (error is System.ComponentModel.Win32Exception or PlatformNotSupportedException)
            {
                Console.WriteLine($"CPU affinity    : unavailable ({error.Message})");
            }
        }
        else
        {
            Console.WriteLine("CPU affinity    : not reported on this platform");
        }

        Console.WriteLine("WARNING: core migration is not ruled out; on Linux run taskset -c <cpu> dotnet run -c Release using an allowed performance CPU.");
    }

    private static bool VerifyTimingStatistics()
    {
        (double[] samples, double median, double firstQuartile, double thirdQuartile)[] cases =
        {
            (new double[] { 100, 4, 1, 3, 2 }, 3, 2, 4),
            (new double[] { 4, 1, 3, 2 }, 2.5, 1.75, 3.25),
            (new double[] { 7, 7, 7, 7 }, 7, 7, 7),
            (new double[] { 5 }, 5, 5, 5),
            (Enumerable.Range(1, TimingStatistics.SampleCount).Select(value => (double)value).Reverse().ToArray(), 16, 8.5, 23.5),
        };

        foreach (var test in cases)
        {
            TimingStatistics actual = TimingStatistics.FromSamples(test.samples);
            if (actual.MedianSeconds != test.median
                || actual.FirstQuartileSeconds != test.firstQuartile
                || actual.ThirdQuartileSeconds != test.thirdQuartile
                || actual.IqrSeconds != test.thirdQuartile - test.firstQuartile)
            {
                Console.WriteLine("Timing checks   : FAIL");
                Environment.ExitCode = 1;
                return false;
            }
        }

        Console.WriteLine("Timing checks   : PASS");
        return true;
    }

    private static bool Verify<TKernel>() where TKernel : struct, IMicroKernel
    {
        using var scratch = GemmScratch.For<TKernel>();
        return Verify((rows, columns, depth, alpha, left, leftStride, right, rightStride, beta, result, resultStride) =>
            Gemm.Multiply<TKernel>(rows, columns, depth, alpha, left, leftStride,
                right, rightStride, beta, result, resultStride, scratch), TKernel.Mr, TKernel.Nr);
    }

    private static bool Verify(MultiplyOperation multiply, int tileRows, int tileColumns)
    {
        // Deliberately ragged shapes so every edge path gets exercised.
        (int rows, int columns, int depth)[] cases =
        {
            (1, 1, 1),
            (tileRows, tileColumns, 1),
            (61, 67, 73),
            (100, 100, 100),
            (129, 33, 257),
            (17, 400, 5),
            (0, 7, 5),
            (7, 0, 5),
            (5, 7, 0),
        };
        (double alpha, double beta)[] scalars = { (1.7, -0.3), (1.0, 0.0), (0.0, 0.5), (1.0, 1.0) };

        double worst = 0.0;
        bool passed = true;

        foreach ((int rows, int columns, int depth) in cases)
            foreach (int padding in new[] { 0, 3 })
                foreach ((double alpha, double beta) in scalars)
                {
                    int leftStride = Math.Max(1, rows) + padding;
                    int rightStride = Math.Max(1, depth) + padding;
                    int resultStride = Math.Max(1, rows) + padding;
                    int resultCount = resultStride * Math.Max(1, columns);
                    double* left = AllocFilled(leftStride * Math.Max(1, depth), seed: 1);
                    double* right = AllocFilled(rightStride * Math.Max(1, columns), seed: 2);
                    double* actual = AllocFilled(resultCount, seed: 3);
                    double* expected = Copy(actual, resultCount);

                    try
                    {
                        if (beta == 0.0)
                            for (int column = 0; column < columns; column++)
                                new Span<double>(actual + column * resultStride, rows).Fill(double.NaN);

                        multiply(rows, columns, depth, alpha, left, leftStride,
                            right, rightStride, beta, actual, resultStride);
                        Reference.Multiply(rows, columns, depth, alpha, left, leftStride,
                            right, rightStride, beta, expected, resultStride);

                        double residual = Reference.RelativeResidual(rows, columns, actual, resultStride, expected, resultStride);
                        worst = Math.Max(worst, residual);

                        bool paddingIntact = true;
                        for (int column = 0; column < Math.Max(1, columns); column++)
                            for (int row = columns == 0 ? 0 : rows; row < resultStride; row++)
                                paddingIntact &= actual[column * resultStride + row] == expected[column * resultStride + row];

                        if (!double.IsFinite(residual) || residual >= 1e-13 || !paddingIntact)
                        {
                            passed = false;
                            Console.WriteLine($"  failed case   : {rows}x{columns}x{depth}, padding={padding}, alpha={alpha}, beta={beta}, residual={residual:E3}, padding intact={paddingIntact}");
                        }
                    }
                    finally
                    {
                        NativeMemory.AlignedFree(left);
                        NativeMemory.AlignedFree(right);
                        NativeMemory.AlignedFree(actual);
                        NativeMemory.AlignedFree(expected);
                    }
                }

        string verdict = passed ? "PASS" : "FAIL";
        Console.WriteLine($"  correctness   : {verdict}  (worst relative residual {worst:E3})");
        if (!passed) Environment.ExitCode = 1;
        return passed;
    }

    private static void Benchmark<TKernel>(double peakGflops) where TKernel : struct, IMicroKernel
    {
        using var scratch = GemmScratch.For<TKernel>();

        Benchmark((rows, columns, depth, alpha, left, leftStride, right, rightStride, beta, result, resultStride) =>
            Gemm.Multiply<TKernel>(rows, columns, depth, alpha, left, leftStride,
                right, rightStride, beta, result, resultStride, scratch), peakGflops);
    }

    private static Dictionary<int, double> Benchmark(MultiplyOperation multiply, double? peakGflops = null, int[]? sizes = null)
    {
        var results = new Dictionary<int, double>();
        Console.WriteLine("     size    median (ms)     IQR (ms)      GFLOP/s   samples"
            + (peakGflops is not null ? "   % of ceiling" : "")
            + (_blisGflops is not null ? "    % of BLIS" : ""));

        foreach (int size in sizes ?? BenchSizes)
        {
            int m = size, n = size, k = size;

            double* a = AllocFilled(m * k, seed: 11);
            double* b = AllocFilled(k * n, seed: 22);
            double* c = AllocFilled(m * n, seed: 33);

            // Warm up: tier up the JIT and warm the caches.
            multiply(m, n, k, 1.0, a, m, b, k, 0.0, c, m);
            multiply(m, n, k, 1.0, a, m, b, k, 0.0, c, m);

            var samples = new double[TimingStatistics.SampleCount];

            for (int sample = 0; sample < samples.Length; sample++)
            {
                long started = Stopwatch.GetTimestamp();
                multiply(m, n, k, 1.0, a, m, b, k, 0.0, c, m);
                samples[sample] = Stopwatch.GetElapsedTime(started).TotalSeconds;
            }

            TimingStatistics timing = TimingStatistics.FromSamples(samples);
            double gflops = 2.0 * m * n * k / timing.MedianSeconds / 1e9;
            results[size] = gflops;

            string ceiling = peakGflops is { } peak ? $"    {100.0 * gflops / peak,10:F1}" : "";
            string blisRatio = _blisGflops is { } baseline ? $"   {100.0 * gflops / baseline[size],10:F1}" : "";
            Console.WriteLine($"  {size,5}     {timing.MedianSeconds * 1e3,10:F3}   {timing.IqrSeconds * 1e3,10:F3}   {gflops,10:F2}   {samples.Length,7}{ceiling}{blisRatio}");

            NativeMemory.AlignedFree(a);
            NativeMemory.AlignedFree(b);
            NativeMemory.AlignedFree(c);
        }

        return results;
    }

    private static void RunParallel<TKernel>(bool verifyOnly) where TKernel : struct, IMicroKernel
    {
        Console.WriteLine();
        Console.WriteLine($"=== {TKernel.Name} threaded (MR={TKernel.Mr}, NR={TKernel.Nr}) ===");

        if (!TKernel.IsSupported)
        {
            Console.WriteLine("  not supported on this CPU, skipping");
            return;
        }

        // Correctness once. Races and edge handling do not depend on thread count,
        // and the default scratch already uses every logical core.
        using (var verifyScratch = ParallelGemmScratch.For<TKernel>())
        {
            if (!Verify((rows, columns, depth, alpha, left, leftStride, right, rightStride, beta, result, resultStride) =>
                    ParallelGemm.Multiply<TKernel>(rows, columns, depth, alpha, left, leftStride,
                        right, rightStride, beta, result, resultStride, verifyScratch),
                    TKernel.Mr, TKernel.Nr)
                || verifyOnly)
            {
                return;
            }
        }

        var byThreadCount = new Dictionary<int, Dictionary<int, double>>();

        foreach (int threads in ThreadCounts())
        {
            Console.WriteLine();
            Console.WriteLine($"  threads = {threads}");

            using var scratch = ParallelGemmScratch.For<TKernel>(threads);

            byThreadCount[threads] = Benchmark(
                (rows, columns, depth, alpha, left, leftStride, right, rightStride, beta, result, resultStride) =>
                    ParallelGemm.Multiply<TKernel>(rows, columns, depth, alpha, left, leftStride,
                        right, rightStride, beta, result, resultStride, scratch));
        }

        PrintScaling(byThreadCount);
    }

    private static int[] ThreadCounts()
    {
        int cores = Environment.ProcessorCount;
        var counts = new SortedSet<int> { 1, cores };

        foreach (int candidate in new[] { 2, 4, 6, 8, 12, 14, 16, 20, 24, 32 })
            if (candidate < cores) counts.Add(candidate);

        return counts.ToArray();
    }

    private static void PrintScaling(Dictionary<int, Dictionary<int, double>> byThreadCount)
    {
        if (!byThreadCount.TryGetValue(1, out Dictionary<int, double>? single)) return;

        Console.WriteLine();
        Console.WriteLine("  scaling vs 1 thread (GFLOP/s and speedup)");
        Console.Write("  threads");
        foreach (int size in BenchSizes) Console.Write($"{size,16}");
        Console.WriteLine();

        foreach ((int threads, Dictionary<int, double> results) in byThreadCount.OrderBy(entry => entry.Key))
        {
            Console.Write($"  {threads,7}");
            foreach (int size in BenchSizes)
                Console.Write($"{results[size],10:F1} {results[size] / single[size],4:F1}x");
            Console.WriteLine();
        }
    }


    private static void BenchmarkReference(int size)
    {
        Console.WriteLine();
        Console.WriteLine("=== naive triple loop (baseline) ===");

        Benchmark(Reference.Multiply, sizes: new[] { size });
    }

    private static double* AllocFilled(int count, int seed)
    {
        double* p = (double*)NativeMemory.AlignedAlloc((nuint)count * sizeof(double), 64);
        var rng = new Random(seed);
        for (int i = 0; i < count; i++) p[i] = rng.NextDouble() - 0.5;
        return p;
    }

    private static double* Copy(double* source, int count)
    {
        double* p = (double*)NativeMemory.AlignedAlloc((nuint)count * sizeof(double), 64);
        Buffer.MemoryCopy(source, p, (long)count * sizeof(double), (long)count * sizeof(double));
        return p;
    }
}
