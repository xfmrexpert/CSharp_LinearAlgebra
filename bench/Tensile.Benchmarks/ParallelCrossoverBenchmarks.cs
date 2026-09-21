using BenchmarkDotNet.Attributes;
using Tensile.Kernels;

namespace Tensile.Benchmarks;

/// <summary>
/// Where threading starts to pay, measured directly rather than derived.
///
/// <c>GemmDispatch.DefaultParallelThreshold</c> is 4e6 multiply-adds, argued
/// from "a fork/join costs tens of microseconds and 4e6 flops is about 160 us
/// of work". Nothing has ever checked it, and it decides serial versus
/// threaded for every trailing update in LU.
///
/// Rather than sweep the threshold -- which would only re-measure these same
/// two paths with a branch in front of them -- this runs both paths explicitly
/// across sizes that bracket it. The crossover in the results IS the measured
/// threshold: read off the smallest shape where "threaded" beats "serial",
/// multiply m*n*k, and that is the value to set.
///
/// Two shape families, because they do not behave alike:
///
/// - **Square**, n from 64 to 384, bracketing 4e6 (n=159 is exactly 4e6).
/// - **Panel**, m = n large with k = 64, which is the shape LU's trailing
///   update actually produces. A panel has the same flop count as a much
///   smaller square but far more memory traffic per flop, so its crossover
///   sits somewhere else, and the threshold LU wants is this one.
/// </summary>
public unsafe class ParallelCrossoverBenchmarks : IDisposable
{
    private double* _a;
    private double* _b;
    private double* _c;

    private GemmScratch? _serial;
    private ParallelGemmScratch? _parallel;

    private int _m;
    private int _k;

    /// <summary>
    /// "n" means a square n x n x n product; "m x k" means an m x m result with
    /// depth k, the shape of an LU trailing update.
    /// </summary>
    public static IEnumerable<string> Shapes =>
    [
        "64", "96", "128", "160", "192", "256", "384",
        "256x64", "512x64", "1024x64", "2048x64",
    ];

    /// <summary>Which shape this row measures.</summary>
    [ParamsSource(nameof(Shapes))]
    public string Shape { get; set; } = "128";

    /// <summary>Parse the shape and allocate operands for it.</summary>
    [GlobalSetup]
    public void Setup()
    {
        int separator = Shape.IndexOf('x', StringComparison.Ordinal);

        if (separator < 0)
        {
            _m = int.Parse(Shape);
            _k = _m;
        }
        else
        {
            _m = int.Parse(Shape[..separator]);
            _k = int.Parse(Shape[(separator + 1)..]);
        }

        _a = GemmBenchmarks.Alloc(_m * _k, seed: 1);
        _b = GemmBenchmarks.Alloc(_k * _m, seed: 2);
        _c = GemmBenchmarks.Alloc(_m * _m, seed: 3);

        _serial = BenchmarkKernel.Scratch(mc: 288, kc: 384, nc: 4096);
        _parallel = BenchmarkKernel.ParallelScratch();
    }

    /// <summary>Release the operands and packing buffers.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>Work in multiply-adds, so the crossover can be read off as a threshold.</summary>
    public long Work => (long)_m * _m * _k;

    /// <summary>The serial driver, whatever the size.</summary>
    [Benchmark(Baseline = true)]
    public void Serial() =>
        BenchmarkKernel.Gemm(_serial!, _m, _m, _k, _a, _m, _b, _k, _c, _m);

    /// <summary>The threaded driver, whatever the size.</summary>
    [Benchmark]
    public void Threaded() =>
        BenchmarkKernel.ParallelGemm(_parallel!, _m, _m, _k, _a, _m, _b, _k, _c, _m);

    /// <summary>Release the operands and packing buffers.</summary>
    public void Dispose()
    {
        if (_a is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_a); _a = null; }
        if (_b is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_b); _b = null; }
        if (_c is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_c); _c = null; }

        _serial?.Dispose();
        _serial = null;

        _parallel?.Dispose();
        _parallel = null;

        GC.SuppressFinalize(this);
    }
}
