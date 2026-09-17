using System.Runtime.InteropServices;
using BenchmarkDotNet.Attributes;
using Tensile.Kernels;

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
/// The kernel is a parameter drawn from <see cref="SupportedKernels"/> rather
/// than one benchmark method each, so a kernel the host cannot run produces no
/// row at all. Methods that returned early on an unsupported CPU were worse
/// than useless: BenchmarkDotNet still measured the empty method and reported
/// an apparently valid ceiling of a few nanoseconds.
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

    /// <summary>Micro-kernels this CPU can actually run.</summary>
    public static IEnumerable<string> SupportedKernels
    {
        get
        {
            if (Avx512Kernel16x8.IsSupported) yield return Avx512Kernel16x8.Name;
            if (Avx2Kernel8x6.IsSupported) yield return Avx2Kernel8x6.Name;
            yield return ScalarKernel4x4.Name;
        }
    }

    /// <summary>Which micro-kernel this run measures.</summary>
    [ParamsSource(nameof(SupportedKernels))]
    public string Kernel { get; set; } = ScalarKernel4x4.Name;

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

    /// <summary>
    /// Run the selected kernel back to back on L1-resident panels. The switch is
    /// per measured operation, not per kernel call, so it costs nothing against
    /// <see cref="Repetitions"/> invocations.
    /// </summary>
    [Benchmark]
    public void Ceiling()
    {
        if (Kernel == Avx512Kernel16x8.Name)
        {
            for (int r = 0; r < Repetitions; r++)
                Avx512Kernel16x8.Execute(Depth, 1.0, _ap, _bp, _c, 16);
        }
        else if (Kernel == Avx2Kernel8x6.Name)
        {
            for (int r = 0; r < Repetitions; r++)
                Avx2Kernel8x6.Execute(Depth, 1.0, _ap, _bp, _c, 8);
        }
        else
        {
            for (int r = 0; r < Repetitions; r++)
                ScalarKernel4x4.Execute(Depth, 1.0, _ap, _bp, _c, 4);
        }
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
