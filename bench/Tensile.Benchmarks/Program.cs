using BenchmarkDotNet.Running;

namespace Tensile.Benchmarks;

/// <summary>Entry point. Run with no arguments for the interactive chooser.</summary>
public static class Program
{
    /// <summary>
    /// Dispatch to BenchmarkDotNet, registering only the benchmarks this
    /// machine can actually run.
    ///
    /// The BLIS comparison is registered conditionally rather than being a
    /// method that no-ops when the library is missing: BenchmarkDotNet measures
    /// whatever it discovers, so a no-op would be reported as a real, absurdly
    /// fast row and would make the managed code look thousands of times slower
    /// than a baseline that never ran.
    /// </summary>
    /// <param name="args">Benchmark filters and BenchmarkDotNet options.</param>
    public static void Main(string[] args)
    {
        var types = new List<Type>
        {
            typeof(GemmBenchmarks),
            typeof(KernelCeilingBenchmarks),
            typeof(LuBenchmarks),
        };

        if (BlisGemmBenchmarks.IsAvailable)
        {
            types.Add(typeof(BlisGemmBenchmarks));
        }
        else
        {
            Console.WriteLine("BLIS not found; the native comparison is omitted from this run.");
            Console.WriteLine("Set TENSILE_BLIS_LIBRARY to a libblis shared library to include it.");
            Console.WriteLine();
        }

        BenchmarkSwitcher.FromTypes([.. types]).Run(args);
    }
}
