using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Tensile.Primitives;

namespace Tensile.Benchmarks;

/// <summary>
/// The micro-kernel in isolation, on packed panels small enough to stay
/// resident in L1, with one C tile held in registers across the whole k-loop.
///
/// This is the kernel's own ceiling: no packing, no cache blocking, no memory
/// traffic beyond streaming two panels. Comparing full GEMM against it
/// separates "the kernel is slow" from "everything around the kernel is slow",
/// which are different problems with different fixes. Full GEMM landing at
/// roughly 88% of this number is the expected, healthy result.
///
/// A kernel the host CPU does not support returns without doing work, so its
/// row is meaningless rather than absent -- read it together with the ISA line
/// that tensile-diag prints.
/// </summary>
public unsafe class KernelCeilingBenchmarks : IDisposable
{
    /// <summary>Panel depth, chosen so both micro-panels stay L1-resident.</summary>
    private const int Depth = 256;

    /// <summary>Kernel calls per measured operation, so one call is not lost in timer noise.</summary>
    private const int Repetitions = 20_000;

    private double* _ap;
    private double* _bp;
    private double* _c;

    /// <summary>Allocate panels sized for the widest kernel; narrower ones use a prefix.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _ap = GemmBenchmarks.Alloc(16 * Depth, seed: 11);
        _bp = GemmBenchmarks.Alloc(8 * Depth, seed: 12);
        _c = GemmBenchmarks.Alloc(16 * 8, seed: 13);
    }

    /// <summary>Release the panels.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>AVX-512 16x8 kernel ceiling.</summary>
    [Benchmark(Description = "AVX-512 16x8")]
    public void Avx512()
    {
        if (!Avx512Kernel16x8.IsSupported) return;

        for (int r = 0; r < Repetitions; r++)
            Avx512Kernel16x8.Execute(Depth, 1.0, _ap, _bp, _c, 16);
    }

    /// <summary>AVX2/FMA 8x6 kernel ceiling.</summary>
    [Benchmark(Description = "AVX2/FMA 8x6")]
    public void Avx2()
    {
        if (!Avx2Kernel8x6.IsSupported) return;

        for (int r = 0; r < Repetitions; r++)
            Avx2Kernel8x6.Execute(Depth, 1.0, _ap, _bp, _c, 8);
    }

    /// <summary>Portable scalar 4x4 kernel ceiling, as a floor to compare against.</summary>
    [Benchmark(Description = "scalar 4x4")]
    public void Scalar()
    {
        for (int r = 0; r < Repetitions; r++)
            ScalarKernel4x4.Execute(Depth, 1.0, _ap, _bp, _c, 4);
    }

    /// <summary>Release the panels.</summary>
    public void Dispose()
    {
        if (_ap is not null) { NativeMemory.AlignedFree(_ap); _ap = null; }
        if (_bp is not null) { NativeMemory.AlignedFree(_bp); _bp = null; }
        if (_c is not null) { NativeMemory.AlignedFree(_c); _c = null; }

        GC.SuppressFinalize(this);
    }
}
