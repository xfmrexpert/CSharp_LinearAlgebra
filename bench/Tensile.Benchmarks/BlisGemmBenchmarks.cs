using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Tensile.Interop;

namespace Tensile.Benchmarks;

/// <summary>
/// Native BLIS as a baseline for the managed GEMM.
///
/// A separate class, registered by <see cref="Program"/> only when the shared
/// library actually loads. The obvious alternative -- one benchmark that does
/// nothing when BLIS is absent -- is worse than useless: BenchmarkDotNet still
/// discovers and measures it, and reports a nanosecond-scale row labelled
/// "BLIS (native)". A missing baseline is obvious; a fake one that makes the
/// managed code look 1000x slower is not.
///
/// Compare against <see cref="GemmBenchmarks"/> at the same N, and check the
/// architecture BLIS dispatched to first: a <c>generic</c> build is a reference
/// fallback and the comparison is meaningless.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public unsafe class BlisGemmBenchmarks : IDisposable
{
    private double* _a;
    private double* _b;
    private double* _c;
    private Blis? _blis;

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
    }

    /// <summary>Release the operands and unload BLIS.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

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

        GC.SuppressFinalize(this);
    }
}
