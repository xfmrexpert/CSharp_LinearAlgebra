using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using Tensile;
using Tensile.Interop;
using Tensile.Kernels;

namespace Tensile.Diagnostics;

/// <summary>
/// Reports that do not belong in either the unit tests or the benchmarks.
///
/// Three jobs, and the first is the one with a hard requirement attached:
///
/// 1. It calls every micro-kernel the host supports, so that
///    <c>disasm.sh</c> can capture their generated code. That gate cannot use
///    the benchmark project, because BenchmarkDotNet runs each benchmark in a
///    generated child process and both processes honour
///    <c>DOTNET_JitStdOutFile</c> — they would clobber each other's output and
///    kernels would silently go missing from the dump, exactly as they did when
///    the dump ran through <c>dotnet run</c>.
/// 2. It prints what the host actually supports and what BLIS dispatched to,
///    without which no measured ratio means anything.
/// 3. It reports estimator accuracy by ensemble, and where a blocked LU
///    spends its time. Those numbers are diagnostic rather than assertable:
///    normest1 is a lower bound, so 38% exact on uniform signed matrices is the
///    ensemble's difficulty and not a defect, and the useful output is the
///    distribution rather than a pass or a fail. The LU phase split is the same
///    kind of thing, and it is here rather than in the benchmarks because
///    BenchmarkDotNet measures a whole call and cannot see inside one.
/// </summary>
public static class Program
{
    /// <summary>Run the reports.</summary>
    /// <param name="args">
    /// <c>--quiet</c> exercises the kernels and prints the environment only,
    /// which is all the codegen gate needs.
    /// </param>
    /// <returns>Zero on success.</returns>
    public static int Main(string[] args)
    {
        bool quiet = args.Contains("--quiet");

        ReportEnvironment();

        // Exercising the kernels also checks them against the reference, and
        // that result is a gate rather than a print: CI runs this step, so a
        // kernel producing NaNs must fail it rather than scroll past.
        bool passed = ExerciseKernels();

        if (!quiet)
        {
            ReportBlis();
            ReportLuPhases();
            ReportEstimatorAccuracy();
        }

        Console.WriteLine(passed ? "kernel checks: PASS" : "kernel checks: FAIL");

        return passed ? 0 : 1;
    }

    private static void ReportEnvironment()
    {
        Console.WriteLine("=== environment ===");
        Console.WriteLine($"  .NET          : {Environment.Version}");
        Console.WriteLine($"  OS            : {RuntimeInformation.OSDescription}");
        Console.WriteLine($"  architecture  : {RuntimeInformation.ProcessArchitecture}");
        Console.WriteLine($"  logical cores : {Environment.ProcessorCount}");
        Console.WriteLine($"  AVX2 / FMA    : {Avx2.IsSupported} / {Fma.IsSupported}");
        Console.WriteLine($"  AVX-512F      : {Avx512F.IsSupported}");
        Console.WriteLine($"  selected kernel: {Workspace.Shared.KernelName}");

        // Environment.ProcessorCount respects the affinity mask, so a pinned
        // process reports the pinned count. Say so rather than let a scaling
        // sweep be quietly read as a machine-wide number.
        Console.WriteLine("  note          : ProcessorCount respects the affinity mask; pin for");
        Console.WriteLine("                  single-thread comparisons, never for scaling sweeps.");
        Console.WriteLine();
    }

    /// <summary>
    /// Run each supported kernel through a real GEMM so the JIT compiles it.
    /// Small sizes on purpose: the goal is compilation, not measurement.
    /// </summary>
    /// <returns>Whether every supported kernel matched the reference.</returns>
    private static unsafe bool ExerciseKernels()
    {
        Console.WriteLine("=== micro-kernels ===");

        bool passed = Exercise<Avx512Kernel16x8>();
        passed &= Exercise<Avx2Kernel8x6>();
        passed &= Exercise<ScalarKernel4x4>();

        Console.WriteLine();
        return passed;
    }

    /// <summary>
    /// Residual bound for a 96x96 product. A correct blocked GEMM lands around
    /// 1e-16; anything near this threshold is a real defect rather than a
    /// summation-order difference.
    /// </summary>
    private const double ResidualLimit = 1e-12;

    private static unsafe bool Exercise<TKernel>() where TKernel : struct, IMicroKernel
    {
        if (!TKernel.IsSupported)
        {
            Console.WriteLine($"  {TKernel.Name,-16}: not supported on this CPU");
            return true;
        }

        const int N = 96;

        using var scratch = GemmScratch.For<TKernel>();

        double* a = Alloc(N * N, seed: 1);
        double* b = Alloc(N * N, seed: 2);
        double* c = Alloc(N * N, seed: 3);
        double* expected = Alloc(N * N, seed: 3);

        try
        {
            Gemm.Multiply<TKernel>(N, N, N, 1.0, a, N, b, N, 0.0, c, N, scratch);
            Reference.Multiply(N, N, N, 1.0, a, N, b, N, 0.0, expected, N);

            double residual = Reference.RelativeResidual(N, N, c, N, expected, N);
            bool ok = double.IsFinite(residual) && residual < ResidualLimit;

            Console.WriteLine(
                $"  {TKernel.Name,-16}: MR={TKernel.Mr} NR={TKernel.Nr}, residual {residual:E3}"
                + (ok ? "" : "   FAIL"));

            return ok;
        }
        finally
        {
            NativeMemory.AlignedFree(a);
            NativeMemory.AlignedFree(b);
            NativeMemory.AlignedFree(c);
            NativeMemory.AlignedFree(expected);
        }
    }

    private static void ReportBlis()
    {
        Console.WriteLine("=== BLIS ===");

        using Blis? blis = Blis.TryLoad(out string reason);

        if (blis is null)
        {
            Console.WriteLine($"  unavailable: {reason}");
            Console.WriteLine();
            return;
        }

        Console.WriteLine($"  library       : {blis.LibraryName} (BLIS {blis.Version}, {blis.IntegerBits}-bit integers)");
        Console.WriteLine($"  architecture  : {blis.Architecture}");
        Console.WriteLine($"  DGEMM kernel  : {blis.GemmKernelImplementation}");
        Console.WriteLine($"  threads       : {blis.Threads}");

        if (blis.Architecture is "generic" or "unknown" || blis.GemmKernelImplementation != "optimized")
        {
            Console.WriteLine("  WARNING: this is a fallback build. Any ratio measured against it is");
            Console.WriteLine("           meaningless; rebuild BLIS with ./configure auto.");
        }

        Console.WriteLine();
    }

    /// <summary>
    /// Where a blocked LU spends its time, by phase.
    ///
    /// This is here because it is the input to an argument rather than a
    /// number in its own right: LU runs at 45% of threaded GEMM, and whether
    /// that is Amdahl or a defect depends entirely on how much of the work is
    /// in the three serial phases (panel, swaps, triangular solve) against the
    /// one threaded phase (the trailing GEMM). CLAUDE.md's reconciliation
    /// rests on this split, so it must be measured on the machine in question
    /// and at the thread count in question, not quoted from elsewhere.
    ///
    /// Read it with the environment in mind. Pinned to one core it reports the
    /// serial split; unpinned it reports the threaded one, and the GEMM share
    /// falls because that is the only phase that gets faster. Both are worth
    /// having and they are different measurements.
    ///
    /// The last column is the honest one: <see cref="LuPhaseTimings.Unattributed"/>
    /// is loop time the four phases do not claim, and it is printed rather than
    /// normalised away.
    /// </summary>
    private static void ReportLuPhases()
    {
        const int Repetitions = 7;

        Console.WriteLine("=== LU phase breakdown ===");
        Console.WriteLine($"  workspace     : {Workspace.Shared.KernelName}, {Environment.ProcessorCount} logical core(s) visible");
        Console.WriteLine("  n     nb    GFLOP/s   panel   swaps    trsm    gemm  unattr    instrument");

        foreach (int n in new[] { 512, 1024, 2048 })
        {
            int nb = Lu.DefaultBlockSizeFor(n, n);

            var a = new Matrix<double>(n, n);

            // Warm up: the first factorization at a size pays for tiering and
            // for first-touch of the packing buffers, neither of which belongs
            // in a phase split.
            Fill(a, seed: n);
            _ = Workspace.Shared.FactorLu(a, nb, timings: null);

            var pooled = new LuPhaseTimings();
            var timedWall = new long[Repetitions];
            var untimedWall = new long[Repetitions];

            // Timed and untimed alternate, and the comparison is between
            // medians, because the interesting quantity is a fraction of a
            // percent and a block-of-one-then-block-of-the-other schedule
            // would charge the whole thermal gradient to the instrument.
            for (int rep = 0; rep < Repetitions; rep++)
            {
                var timings = new LuPhaseTimings();

                Fill(a, seed: n + rep);
                long start = Stopwatch.GetTimestamp();
                _ = Workspace.Shared.FactorLu(a, nb, timings);
                timedWall[rep] = Stopwatch.GetTimestamp() - start;

                pooled.Add(timings);

                Fill(a, seed: n + rep);
                start = Stopwatch.GetTimestamp();
                _ = Workspace.Shared.FactorLu(a, nb, timings: null);
                untimedWall[rep] = Stopwatch.GetTimestamp() - start;
            }

            double seconds = (double)pooled.Total / Stopwatch.Frequency;
            double flops = ((2.0 / 3.0) * n - 0.5) * n * n - n / 6.0;
            double gflops = seconds == 0.0 ? 0.0 : Repetitions * flops / seconds / 1e9;

            double timed = Median(timedWall);
            double untimed = Median(untimedWall);
            double instrument = untimed == 0.0 ? 0.0 : 100.0 * (timed - untimed) / untimed;

            Console.WriteLine(
                $"  {n,-5} {nb,-4} {gflops,8:F2}  {pooled.Share(pooled.Panel),5:F1}%  "
                + $"{pooled.Share(pooled.Swaps),5:F1}%  {pooled.Share(pooled.Triangular),5:F1}%  "
                + $"{pooled.Share(pooled.Gemm),5:F1}%  {pooled.Share(pooled.Unattributed),5:F1}%  "
                + $"{instrument,8:+0.00;-0.00;0.00}%");
        }

        Console.WriteLine("  note          : pinned gives the serial split, unpinned the threaded one.");
        Console.WriteLine("                  The instrument column is the cost of collecting, and is");
        Console.WriteLine("                  noise rather than overhead if it straddles zero.");
        Console.WriteLine();

        static double Median(long[] samples)
        {
            long[] sorted = (long[])samples.Clone();
            Array.Sort(sorted);

            return sorted.Length % 2 == 1
                ? sorted[sorted.Length / 2]
                : 0.5 * (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]);
        }

        static void Fill(Matrix<double> a, int seed)
        {
            var rng = new Random(seed);
            int n = a.Rows;

            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    a[i, j] = rng.NextDouble() - 0.5;

            // Diagonally dominant, so the factorization is well conditioned and
            // the pivot search does not become the story.
            for (int i = 0; i < n; i++) a[i, i] += n;
        }
    }

    /// <summary>
    /// How close the 1-norm estimator gets, split by ensemble.
    ///
    /// Non-negative matrices are the case where the algorithm is provably
    /// sharp — the first sign matrix is all ones, so A^T*S is exactly the
    /// vector of column sums and the first sort lands on the true maximiser.
    /// Anything below 100% there is a bug. The signed rows are the hard end,
    /// where column norms cluster and many near-ties compete.
    /// </summary>
    private static unsafe void ReportEstimatorAccuracy()
    {
        Console.WriteLine("=== normest1 accuracy ===");
        Console.WriteLine("  ensemble          exact      worst ratio");

        Report("non-negative", nonNegative: true, skewed: false, columns: 2);
        Report("signed, t=2", nonNegative: false, skewed: false, columns: 2);
        Report("signed, t=4", nonNegative: false, skewed: false, columns: 4);

        // With a few dominant columns the maximiser stands out and the
        // estimator finds it far more often. This is the realistic case; the
        // uniform rows above are the adversarial one, where column norms
        // cluster within a few percent and many near-ties compete.
        Report("skewed, t=2", nonNegative: false, skewed: true, columns: 2);
        Report("skewed, t=4", nonNegative: false, skewed: true, columns: 4);

        Console.WriteLine();

        static void Report(string label, bool nonNegative, bool skewed, int columns)
        {
            int[] sizes = [4, 9, 16, 33, 64, 129, 256];
            int cases = 0;
            int exact = 0;
            double worst = double.PositiveInfinity;

            foreach (int n in sizes)
            {
                for (int trial = 0; trial < 40; trial++)
                {
                    var a = new Matrix<double>(n, n);
                    var rng = new Random(n * 1000 + trial);

                    for (int j = 0; j < n; j++)
                    {
                        double weight = skewed && j % 7 == 0 ? 10.0 : 1.0;

                        for (int i = 0; i < n; i++)
                            a[i, j] = weight * (nonNegative ? rng.NextDouble() : rng.NextDouble() - 0.5);
                    }

                    double truth = a.OneNorm();
                    double estimate = a.EstimateOneNorm(columns: columns);

                    cases++;
                    if (estimate >= truth * (1.0 - 1e-13)) exact++;
                    worst = Math.Min(worst, estimate / truth);
                }
            }

            Console.WriteLine($"  {label,-16}  {exact,4}/{cases} ({(double)exact / cases,6:P1})   {worst:F4}");
        }
    }

    private static unsafe double* Alloc(int count, int seed)
    {
        var rng = new Random(seed);
        var data = (double*)NativeMemory.AlignedAlloc((nuint)count * sizeof(double), 64);

        for (int i = 0; i < count; i++) data[i] = rng.NextDouble() - 0.5;

        return data;
    }
}
