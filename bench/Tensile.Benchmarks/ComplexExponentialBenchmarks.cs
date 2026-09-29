using System.Numerics;
using BenchmarkDotNet.Attributes;

namespace Tensile.Benchmarks;

/// <summary>
/// The native complex exponentials against the embedded route they replaced,
/// **both rows in one class** so BenchmarkDotNet interleaves them over the
/// same operands (finding 12).
///
/// What to expect, from arithmetic rather than measurement: native <c>Expm</c>
/// does half the flops of the embedded one -- a 4M product of order N is 8N^3
/// real flops against 16N^3 for a real product of order 2N -- and may take
/// one squaring fewer, since it reads complex norms rather than real ones up
/// to sqrt(2) larger. Native <c>Expmv</c> does the same flops as the embedded
/// one (an embedded matrix times a stacked panel is not redundant), so the
/// comparison there is memory traffic: the embedded route streams a real
/// matrix of twice the bytes per application. Neither prediction has been
/// measured.
///
/// Single-threaded workspace and pinned (<c>taskset -c 0</c>): efficiency,
/// not scaling. The operand's 1-norm is fixed at <see cref="Norm"/>, so the
/// scaling choice, and with it the product count, is comparable across N.
/// Pass <c>--iterationCount 31</c>.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class ComplexExponentialBenchmarks : IDisposable
{
    /// <summary>The 1-norm every operand is scaled to.</summary>
    private const double Norm = 10.0;

    private Workspace? _workspace;
    private Matrix<Complex>? _a;
    private Matrix<Complex>? _b;

    /// <summary>Order of the complex operand.</summary>
    [Params(64, 256)]
    public int N { get; set; }

    /// <summary>A random complex operand of 1-norm <see cref="Norm"/>, and a one-column B.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _workspace = new Workspace(multithreaded: false);
        var rng = new Random(20260929);

        _a = new Matrix<Complex>(N, N);
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                _a[i, j] = new Complex(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5);

        double scale = Norm / _a.OneNorm();
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                _a[i, j] *= scale;

        _b = new Matrix<Complex>(N, 1);
        for (int i = 0; i < N; i++) _b[i, 0] = new Complex(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5);
    }

    /// <summary>exp(A), native: the baseline.</summary>
    [Benchmark(Baseline = true)]
    public Matrix<Complex> ExpmNative() => _a!.Expm(_workspace);

    /// <summary>exp(A) through the 2N x 2N real representation.</summary>
    [Benchmark]
    public Matrix<Complex> ExpmEmbedded() => ComplexEmbedding.Expm(_a!, _workspace);

    /// <summary>exp(A)b, native.</summary>
    [Benchmark]
    public Matrix<Complex> ExpmvNative() => _a!.Expmv(_b!.ReadOnlyView);

    /// <summary>exp(A)b through the real representation.</summary>
    [Benchmark]
    public Matrix<Complex> ExpmvEmbedded() => ComplexEmbedding.Expmv(_a!, _b!.ReadOnlyView, 1.0);

    /// <summary>Release the workspace.</summary>
    [GlobalCleanup]
    public void Dispose()
    {
        _workspace?.Dispose();
        _workspace = null;
        GC.SuppressFinalize(this);
    }
}
