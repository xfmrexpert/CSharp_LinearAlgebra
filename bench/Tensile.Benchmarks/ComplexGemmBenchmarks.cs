using System.Numerics;
using BenchmarkDotNet.Attributes;

namespace Tensile.Benchmarks;

/// <summary>
/// How efficient the 4M complex product is, against the two things it has to
/// be judged by, **all three rows in one class** so BenchmarkDotNet interleaves
/// them in one process (finding 12: a ratio is only as good as the
/// interleaving behind it).
///
/// - <see cref="RealGemm"/> is the unit: one real product of order N. 4M is
///   four of them plus splitting the operands and combining the result, so
///   <c>Complex4M / RealGemm</c> near 4.0 means 4M runs at the real kernel's
///   efficiency, and the excess over 4.0 is what the split and combine cost.
///   That excess is the number that decides whether BLIS's 1M method --
///   splitting inside the packing instead of into temporaries -- is worth
///   writing. The workspace retains the split buffers up to 2^21 elements,
///   so at N=128 and N=512 the excess is copying alone and the allocation
///   column reads zero; at N=1024 the six buffers (6.3M elements) are over
///   the cap and are allocated per call, so that row includes allocation.
/// - <see cref="Complex4M"/> is the product under test, through the public
///   API.
/// - <see cref="EmbeddedReal"/> is the real product of order 2N that the
///   real-embedding route does for the same complex product: eight units of
///   work, so it should read near 2x <see cref="Complex4M"/>. It is what
///   complex <c>Expm</c> pays today for every product, until the native path
///   replaces it.
///
/// Single-threaded workspace and pinned (<c>taskset -c 0</c>): this measures
/// efficiency, not scaling. Pass <c>--iterationCount 31</c>.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class ComplexGemmBenchmarks : IDisposable
{
    private Workspace? _workspace;

    private Matrix<double>? _ra, _rb, _rc;
    private Matrix<Complex>? _za, _zb, _zc;
    private Matrix<double>? _ea, _eb, _ec;

    /// <summary>Square complex problem size; all three dimensions are N.</summary>
    [Params(128, 512, 1024)]
    public int N { get; set; }

    /// <summary>Allocate every operand once per parameter set.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _workspace = new Workspace(multithreaded: false);
        var rng = new Random(20260926);

        _ra = Real(N, rng); _rb = Real(N, rng); _rc = new Matrix<double>(N, N);
        _za = ComplexOf(N, rng); _zb = ComplexOf(N, rng); _zc = new Matrix<Complex>(N, N);
        _ea = Real(2 * N, rng); _eb = Real(2 * N, rng); _ec = new Matrix<double>(2 * N, 2 * N);
    }

    /// <summary>One real product of order N: the unit the other two are measured in.</summary>
    [Benchmark(Baseline = true)]
    public void RealGemm() => _workspace!.Multiply(_ra!.ReadOnlyView, _rb!.ReadOnlyView, _rc!.View);

    /// <summary>The complex product of order N by 4M: four units of real work, plus the split and combine.</summary>
    [Benchmark]
    public void Complex4M() => _workspace!.Multiply(_za!.ReadOnlyView, _zb!.ReadOnlyView, _zc!.View);

    /// <summary>The real product of order 2N the embedding route does instead: eight units.</summary>
    [Benchmark]
    public void EmbeddedReal() => _workspace!.Multiply(_ea!.ReadOnlyView, _eb!.ReadOnlyView, _ec!.View);

    /// <summary>Release the workspace.</summary>
    [GlobalCleanup]
    public void Dispose()
    {
        _workspace?.Dispose();
        _workspace = null;
        GC.SuppressFinalize(this);
    }

    private static Matrix<double> Real(int n, Random rng)
    {
        var m = new Matrix<double>(n, n);
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) m[i, j] = rng.NextDouble() - 0.5;
        return m;
    }

    private static Matrix<Complex> ComplexOf(int n, Random rng)
    {
        var m = new Matrix<Complex>(n, n);
        for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) m[i, j] = new Complex(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5);
        return m;
    }
}
