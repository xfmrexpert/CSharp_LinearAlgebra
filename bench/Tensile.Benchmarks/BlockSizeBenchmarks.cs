using BenchmarkDotNet.Attributes;
using Tensile.Kernels;

namespace Tensile.Benchmarks;

/// <summary>
/// Cache-blocking parameters for the SERIAL GEMM path.
///
/// <c>Gemm.cs</c> still carries the placeholder MC=288, KC=384 from the first
/// commit, described there as "not tuned for any particular machine". The
/// threaded path has since moved to cache-derived values -- KC=256 so one A
/// plus one B micro-panel fit a Gracemont E-core's 32 KiB L1, MC=144 so the
/// packed A block fits an E-core's quarter share of its cluster's L2 -- and
/// those measured better on the 12700H at every size below 2048. Whether they
/// also win on the serial path, where there is no E-core to size for and the
/// whole L2 belongs to one thread, has never been checked.
///
/// KC was a parameter here and is no longer: 256 and 384 came out within 1% of
/// each other at every size in both sweep directions, so the cross term is
/// answered and spending half the run on it again would be churn. Re-add
/// <c>256</c> to its Params if a kernel change makes it live again. NC stays at
/// 4096 throughout; it sizes the packed B block against L3 and is a different
/// question, never varied.
///
/// **Both the driver and the dispatch are measured, and that is the point of
/// the second method.** This class calls <c>Gemm.Multiply</c> directly;
/// <c>GemmBenchmarks</c> goes through <c>GemmDispatch</c>, which is what every
/// caller of the library takes. The two have disagreed by 8-18% on identical
/// configuration across sittings, and the MC question now depends on which of
/// them you believe: the driver sweep put MC=144 ahead by 9.5% at n=2048 in
/// both directions, while the pinned dispatch run that followed it lost 10
/// points of ratio against BLIS at n>=1024 after MC=144 shipped. Those cannot
/// both be right about the shipped path. Measuring them side by side in one
/// class removes the only explanation that does not implicate either -- that
/// the gap is drift between two processes run at different times.
///
/// Pin this one: <c>taskset -c 0</c>. It is a single-threaded measurement, and
/// an unpinned run lets the scheduler migrate it across a cache boundary
/// mid-sample, which is exactly the variable under test.
/// </summary>
public unsafe class BlockSizeBenchmarks : IDisposable
{
    private double* _a;
    private double* _b;
    private double* _c;

    private GemmScratch? _scratch;
    private GemmDispatch? _dispatch;

    /// <summary>Row-block size: the cache-derived 144 against the placeholder 288.</summary>
    [Params(144, 288)]
    public int Mc { get; set; }

    /// <summary>
    /// K-slab depth. Settled at 384: the cache-derived 256 measured within 1%
    /// of it everywhere in both directions, so it is held fixed rather than
    /// doubling the run for a known non-result.
    /// </summary>
    [Params(384)]
    public int Kc { get; set; }

    /// <summary>Square problem size; all three dimensions are N.</summary>
    [Params(128, 512, 2048)]
    public int N { get; set; }

    /// <summary>Allocate operands and a scratch with this row's block sizes.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _a = GemmBenchmarks.Alloc(N * N, seed: 1);
        _b = GemmBenchmarks.Alloc(N * N, seed: 2);
        _c = GemmBenchmarks.Alloc(N * N, seed: 3);

        _scratch = BenchmarkKernel.Scratch(Mc, Kc, nc: 4096);

        // A second scratch, because the dispatch takes ownership of the one it
        // is given and the two methods must not share packing buffers.
        _dispatch = BenchmarkKernel.SerialWith(BenchmarkKernel.Scratch(Mc, Kc, nc: 4096));
    }

    /// <summary>Release the operands and packing buffers.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>The serial driver with the block sizes under test.</summary>
    [Benchmark(Baseline = true)]
    public void Driver() =>
        BenchmarkKernel.Gemm(_scratch!, N, N, N, _a, N, _b, N, _c, N);

    /// <summary>
    /// The same product at the same block sizes, through the dispatch every
    /// caller of the library goes through. Against the baseline this is the
    /// cost of the dispatch layer measured inside one class, where it cannot
    /// be drift between processes.
    /// </summary>
    [Benchmark]
    public void Dispatch() =>
        BenchmarkKernel.Multiply(_dispatch!, N, N, N, _a, N, _b, N, _c, N);

    /// <summary>Release the operands and packing buffers.</summary>
    public void Dispose()
    {
        if (_a is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_a); _a = null; }
        if (_b is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_b); _b = null; }
        if (_c is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_c); _c = null; }

        _scratch?.Dispose();
        _scratch = null;

        _dispatch?.Dispose();
        _dispatch = null;

        GC.SuppressFinalize(this);
    }
}
