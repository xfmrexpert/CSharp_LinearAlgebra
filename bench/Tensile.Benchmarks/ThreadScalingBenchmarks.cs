using BenchmarkDotNet.Attributes;
using Tensile.Kernels;

namespace Tensile.Benchmarks;

/// <summary>
/// How the threaded GEMM scales with worker count.
///
/// Three rules govern reading this table, and each was paid for once:
///
/// - **Do not pin this run.** Environment.ProcessorCount respects the affinity
///   mask, so <c>taskset -c 0</c> makes every row report one thread and the
///   sweep degenerates. Pin the single-thread comparison, never this.
/// - **Run it in both orders.** Consecutive configurations on a laptop part
///   run progressively heat-soaked, so a monotonically rising thread count
///   confounds scaling with thermal state. Set <c>TENSILE_BENCH_REVERSE=1</c>
///   to execute the cases back to front and compare: if the two orders
///   disagree, the sweep measured the cooler, not the code. This is the
///   decisive test CLAUDE.md finding 7 asks for and had never been run. The
///   reversal is done by <see cref="ThermalOrderer"/>, because reversing the
///   parameter list below would not have worked -- BenchmarkDotNet sorts cases
///   by parameter value before it runs them.
/// - **Capture turbostat alongside.** A per-thread efficiency that falls off
///   while package power is pinned at its limit is a power result, not a
///   scheduling one.
///
/// One caveat on the small size. ParallelGemm scales its own worker count down
/// for small problems -- <c>Clamp(work / 8e6, 1, MaxThreads)</c> -- so at
/// N=512 the effective count is capped near 16 however many this sweep asks
/// for, and rows beyond that repeat. At N=2048 the clamp is inactive and the
/// scaling is the parallel structure's own.
/// </summary>
public unsafe class ThreadScalingBenchmarks : IDisposable
{
    private double* _a;
    private double* _b;
    private double* _c;

    private ParallelGemmScratch? _scratch;

    /// <summary>
    /// Worker counts to sweep: the usual powers and half-steps up to this
    /// machine's logical core count.
    /// </summary>
    public static IEnumerable<int> ThreadCounts
    {
        get
        {
            int cores = Environment.ProcessorCount;
            int[] candidates = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64];

            return candidates.Where(c => c < cores).Append(cores).Distinct();
        }
    }

    /// <summary>Worker-count cap for this row.</summary>
    [ParamsSource(nameof(ThreadCounts))]
    public int Threads { get; set; } = 1;

    /// <summary>Square problem size; all three dimensions are N.</summary>
    [Params(512, 2048)]
    public int N { get; set; }

    /// <summary>Allocate the operands and a scratch capped at this row's thread count.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _a = GemmBenchmarks.Alloc(N * N, seed: 1);
        _b = GemmBenchmarks.Alloc(N * N, seed: 2);
        _c = GemmBenchmarks.Alloc(N * N, seed: 3);

        _scratch = BenchmarkKernel.ParallelScratch(Threads);
    }

    /// <summary>Release the operands and packing buffers.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>
    /// The threaded driver directly, not through the dispatch, so the
    /// size threshold cannot silently send a row down the serial path.
    /// </summary>
    [Benchmark]
    public void Threaded() =>
        BenchmarkKernel.ParallelGemm(_scratch!, N, N, N, _a, N, _b, N, _c, N);

    /// <summary>Release the operands and packing buffers.</summary>
    public void Dispose()
    {
        if (_a is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_a); _a = null; }
        if (_b is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_b); _b = null; }
        if (_c is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_c); _c = null; }

        _scratch?.Dispose();
        _scratch = null;

        GC.SuppressFinalize(this);
    }
}
