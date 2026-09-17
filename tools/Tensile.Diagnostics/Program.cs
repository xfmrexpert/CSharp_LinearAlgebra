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
/// 3. It reports estimator accuracy by ensemble. Those numbers are diagnostic
///    rather than assertable: normest1 is a lower bound, so 38% exact on
///    uniform signed matrices is the ensemble's difficulty and not a defect,
///    and the useful output is the distribution rather than a pass or a fail.
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
