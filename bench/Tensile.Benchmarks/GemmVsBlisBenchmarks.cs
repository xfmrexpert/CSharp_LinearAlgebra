using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Tensile.Interop;
using Tensile.Kernels;

namespace Tensile.Benchmarks;

/// <summary>
/// The headline comparison: managed GEMM against native BLIS, **both rows in
/// one class** so BenchmarkDotNet interleaves them in one process.
///
/// That last part is not a detail, it is the whole reason this class has a
/// Tensile row at all. The two used to be measured in separate classes and the
/// ratio computed across the two reports, and separate classes are separate
/// processes run minutes apart. On this machine that cost two conclusions: an
/// 8-18% "gap" between two ways of calling the same GEMM, which turned out not
/// to exist when both were measured in one class; and a 10-12 point drop in
/// the ratio against BLIS at n&gt;=1024, blamed on a block-size change that a
/// within-class A/B then showed to be a wash. Both were drift between
/// processes. A ratio worth quoting has to come from rows BDN ran back to back
/// against the same operands.
///
/// Registered by <see cref="Program"/> only when the shared library actually
/// loads. The obvious alternative -- one benchmark that does nothing when BLIS
/// is absent -- is worse than useless: BenchmarkDotNet still discovers and
/// measures it, and reports a nanosecond-scale row labelled "BLIS (native)".
/// A missing baseline is obvious; a fake one that makes the managed code look
/// 1000x slower is not.
///
/// Pin it (<c>taskset -c 0</c>): this is the single-core comparison. Check the
/// architecture BLIS dispatched to first -- a <c>generic</c> build is a
/// reference fallback and the comparison is meaningless. And pass
/// <c>--iterationCount 31</c>, because the effects being chased here are in
/// single-digit percent and BDN's default job does not resolve them.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public unsafe class GemmVsBlisBenchmarks : IDisposable
{
    private double* _a;
    private double* _b;
    private double* _c;
    private Blis? _blis;
    private GemmDispatch? _serial;

    /// <summary>Square problem size; all three dimensions are N.</summary>
    [Params(128, 256, 512, 1024, 2048)]
    public int N { get; set; }

    /// <summary>Allocate the operands and load BLIS.</summary>
    /// <exception cref="InvalidOperationException">BLIS could not be loaded.</exception>
    [GlobalSetup]
    public void Setup()
    {
        _blis = Blis.TryLoad(out string reason);

        // Program only registers this class when BLIS is available, so reaching
        // here without it means something changed underneath us. Fail loudly
        // rather than measure nothing.
        if (_blis is null) throw new InvalidOperationException(reason);

        _a = GemmBenchmarks.Alloc(N * N, seed: 1);
        _b = GemmBenchmarks.Alloc(N * N, seed: 2);
        _c = GemmBenchmarks.Alloc(N * N, seed: 3);

        _serial = BenchmarkKernel.Serial();
    }

    /// <summary>Release the operands and unload BLIS.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>The managed GEMM, single-threaded, over the same operands.</summary>
    [Benchmark(Baseline = true, Description = "Tensile, 1 thread")]
    public void Managed() =>
        BenchmarkKernel.Multiply(_serial!, N, N, N, _a, N, _b, N, _c, N);

    /// <summary>Native BLIS dgemm.</summary>
    [Benchmark(Description = "BLIS (native)")]
    public void Native() => _blis!.Multiply(N, N, N, 1.0, _a, N, _b, N, 0.0, _c, N);

    /// <summary>Whether a BLIS shared library can be loaded on this machine.</summary>
    public static bool IsAvailable
    {
        get
        {
            using Blis? blis = Blis.TryLoad(out _);
            return blis is not null;
        }
    }

    /// <summary>Release the operands and unload BLIS.</summary>
    public void Dispose()
    {
        if (_a is not null) { NativeMemory.AlignedFree(_a); _a = null; }
        if (_b is not null) { NativeMemory.AlignedFree(_b); _b = null; }
        if (_c is not null) { NativeMemory.AlignedFree(_c); _c = null; }

        _blis?.Dispose();
        _blis = null;

        _serial?.Dispose();
        _serial = null;

        GC.SuppressFinalize(this);
    }
}
