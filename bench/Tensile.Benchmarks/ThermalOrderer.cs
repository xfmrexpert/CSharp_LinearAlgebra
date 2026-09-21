using System.Collections.Immutable;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Order;
using BenchmarkDotNet.Running;

namespace Tensile.Benchmarks;

/// <summary>
/// Runs the benchmark cases in reverse when <c>TENSILE_BENCH_REVERSE=1</c> is
/// set, so a sweep can be executed in both directions.
///
/// This exists because of CLAUDE.md finding 7: on a laptop part, consecutive
/// configurations run progressively heat-soaked, so a sweep that walks a
/// parameter upward confounds the parameter with thermal state. Running it
/// downward and comparing is the cheap decisive test.
///
/// BenchmarkDotNet sorts cases by parameter value before executing them, and
/// it does so regardless of the order a ParamsSource yields — reversing the
/// source list changes the summary not at all and the execution order not at
/// all. That was measured, not assumed: an earlier version of this sweep
/// reversed its parameter list, and the "// Benchmark:" lines in the two logs
/// came out identical. Overriding the execution order is the only thing that
/// actually works.
/// </summary>
public sealed class ThermalOrderer : DefaultOrderer
{
    /// <summary>Whether the environment has asked for a reversed run.</summary>
    public static bool Reversed =>
        Environment.GetEnvironmentVariable("TENSILE_BENCH_REVERSE") == "1";

    /// <inheritdoc/>
    public override IEnumerable<BenchmarkCase> GetExecutionOrder(
        ImmutableArray<BenchmarkCase> benchmarkCases,
        IEnumerable<BenchmarkLogicalGroupRule>? order = null)
    {
        IEnumerable<BenchmarkCase> ordered = base.GetExecutionOrder(benchmarkCases, order);

        return Reversed ? ordered.Reverse() : ordered;
    }
}
