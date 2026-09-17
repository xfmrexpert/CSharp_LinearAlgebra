using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Tensile.Kernels;

namespace Tensile.Benchmarks;

/// <summary>
/// GEMM throughput, managed against native BLIS.
///
/// The methodology that used to live in the hand-rolled harness is now
/// BenchmarkDotNet's job: warmup, tiering, process isolation and the statistics.
/// What does not carry over automatically is the discipline around it, so:
///
/// - pin with <c>taskset -c 0</c> for the single-threaded comparison, and do
///   NOT pin for the threaded one, because Environment.ProcessorCount respects
///   the affinity mask and a pinned scaling sweep is meaningless;
/// - check the BLIS architecture that <c>tensile-diag</c> prints before quoting
///   any ratio, since a <c>generic</c> BLIS build is a reference fallback and
///   not a competitor;
/// - on a laptop part, capture <c>turbostat</c> alongside, because sustained
///   multi-threaded runs are usually power-limited rather than
///   algorithm-limited.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public unsafe class GemmBenchmarks : IDisposable
{
    private double* _a;
    private double* _b;
    private double* _c;

    private GemmDispatch? _serial;
    private GemmDispatch? _parallel;

    /// <summary>Square problem size; all three dimensions are N.</summary>
    [Params(128, 256, 512, 1024, 2048)]
    public int N { get; set; }

    /// <summary>Allocate and fill the operands once per parameter set.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _a = Alloc(N * N, seed: 1);
        _b = Alloc(N * N, seed: 2);
        _c = Alloc(N * N, seed: 3);

        _serial = CreateDispatch(multithreaded: false);
        _parallel = CreateDispatch(multithreaded: true);
    }

    /// <summary>Release the operands and packing buffers.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>Single-threaded blocked GEMM on the widest supported kernel.</summary>
    [Benchmark(Baseline = true, Description = "Tensile, 1 thread")]
    public void Serial() => Multiply(_serial!);

    /// <summary>Multi-threaded blocked GEMM on the widest supported kernel.</summary>
    [Benchmark(Description = "Tensile, threaded")]
    public void Parallel() => Multiply(_parallel!);

    private void Multiply(GemmDispatch dispatch)
    {
        if (Avx512Kernel16x8.IsSupported)
            dispatch.Multiply<Avx512Kernel16x8>(N, N, N, 1.0, _a, N, _b, N, 0.0, _c, N);
        else if (Avx2Kernel8x6.IsSupported)
            dispatch.Multiply<Avx2Kernel8x6>(N, N, N, 1.0, _a, N, _b, N, 0.0, _c, N);
        else
            dispatch.Multiply<ScalarKernel4x4>(N, N, N, 1.0, _a, N, _b, N, 0.0, _c, N);
    }

    private static GemmDispatch CreateDispatch(bool multithreaded)
    {
        if (Avx512Kernel16x8.IsSupported)
            return multithreaded
                ? GemmDispatch.Multithreaded<Avx512Kernel16x8>()
                : GemmDispatch.Serial<Avx512Kernel16x8>();

        if (Avx2Kernel8x6.IsSupported)
            return multithreaded
                ? GemmDispatch.Multithreaded<Avx2Kernel8x6>()
                : GemmDispatch.Serial<Avx2Kernel8x6>();

        return multithreaded
            ? GemmDispatch.Multithreaded<ScalarKernel4x4>()
            : GemmDispatch.Serial<ScalarKernel4x4>();
    }

    internal static double* Alloc(int count, int seed)
    {
        var rng = new Random(seed);
        var data = (double*)NativeMemory.AlignedAlloc((nuint)count * sizeof(double), 64);

        for (int i = 0; i < count; i++) data[i] = rng.NextDouble() - 0.5;

        return data;
    }

    /// <summary>Release the operands and packing buffers.</summary>
    public void Dispose()
    {
        if (_a is not null) { NativeMemory.AlignedFree(_a); _a = null; }
        if (_b is not null) { NativeMemory.AlignedFree(_b); _b = null; }
        if (_c is not null) { NativeMemory.AlignedFree(_c); _c = null; }

        _serial?.Dispose();
        _serial = null;

        _parallel?.Dispose();
        _parallel = null;

        GC.SuppressFinalize(this);
    }
}
