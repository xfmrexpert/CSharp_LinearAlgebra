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
/// The cross terms are deliberate: MC and KC trade against each other, so
/// (144, 384) and (288, 256) have to be in the table or the conclusion is
/// "one of these pairs is better" rather than "this is the better MC and this
/// is the better KC". NC stays at 4096 throughout; it sizes the packed B block
/// against L3 and is a different question.
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

    /// <summary>Row-block size: the cache-derived 144 against the placeholder 288.</summary>
    [Params(144, 288)]
    public int Mc { get; set; }

    /// <summary>K-slab depth: the cache-derived 256 against the placeholder 384.</summary>
    [Params(256, 384)]
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
    }

    /// <summary>Release the operands and packing buffers.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>The serial driver with the block sizes under test.</summary>
    [Benchmark]
    public void Serial() =>
        BenchmarkKernel.Gemm(_scratch!, N, N, N, _a, N, _b, N, _c, N);

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
