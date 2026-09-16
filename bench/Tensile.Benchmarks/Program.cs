using BenchmarkDotNet.Running;

namespace Tensile.Benchmarks;

/// <summary>Entry point. Run with no arguments for the interactive chooser.</summary>
public static class Program
{
    /// <summary>Dispatch to BenchmarkDotNet's switcher.</summary>
    /// <param name="args">Benchmark filters and BenchmarkDotNet options.</param>
    public static void Main(string[] args) =>
        BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
}
