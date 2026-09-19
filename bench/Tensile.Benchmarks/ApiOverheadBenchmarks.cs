using BenchmarkDotNet.Attributes;
using Tensile.Kernels;

namespace Tensile.Benchmarks;

/// <summary>
/// What the secure-by-design migration cost, measured rather than assumed.
///
/// This is the guardrail docs/security-design.md section 9 asks for, and the
/// one number the whole migration rests on. The other GEMM benchmarks allocate
/// with NativeMemory and call the dispatch directly, so they measure the
/// kernels and skip everything the migration added. These three rows walk the
/// difference one layer at a time:
///
/// - <see cref="Native"/> is the old world and the baseline: 64-byte aligned
///   native memory, straight into the dispatch.
/// - <see cref="Pinned"/> changes only the storage -- a Matrix&lt;double&gt; on
///   the pinned object heap, whose first element is aligned by over-allocating
///   a cache line -- and still calls the dispatch directly. The gap against the
///   baseline is the cost of POH storage and of whatever alignment it lands on.
/// - <see cref="PublicApi"/> is what a caller actually runs:
///   Workspace.Multiply over bound views. On top of Pinned it adds Bind, the
///   Operand length checks, the workspace lock and the pinning seam.
///
/// All three are O(1) per call against O(n^3) work, so the ratio should
/// approach 1 as N grows and any real cost shows up at N=128. A ratio that
/// does not converge means something is being done per element rather than per
/// call, which would be a defect and not a tax.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public unsafe class ApiOverheadBenchmarks : IDisposable
{
    private double* _a;
    private double* _b;
    private double* _c;

    private Matrix<double>? _ma;
    private Matrix<double>? _mb;
    private Matrix<double>? _mc;

    private GemmDispatch? _dispatch;
    private Workspace? _workspace;

    /// <summary>Square problem size; all three dimensions are N.</summary>
    [Params(128, 512, 2048)]
    public int N { get; set; }

    /// <summary>Allocate both sets of operands once per parameter set.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _a = GemmBenchmarks.Alloc(N * N, seed: 1);
        _b = GemmBenchmarks.Alloc(N * N, seed: 2);
        _c = GemmBenchmarks.Alloc(N * N, seed: 3);

        _ma = Fill(seed: 1);
        _mb = Fill(seed: 2);
        _mc = Fill(seed: 3);

        _dispatch = BenchmarkKernel.Multithreaded();
        _workspace = new Workspace(multithreaded: true);
    }

    /// <summary>Release the operands, the packing buffers and the workspace.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>Native aligned memory straight into the dispatch: the pre-migration path.</summary>
    [Benchmark(Baseline = true, Description = "native pointers")]
    public void Native() =>
        BenchmarkKernel.Multiply(_dispatch!, N, N, N, _a, N, _b, N, _c, N);

    /// <summary>Pinned-object-heap storage, still straight into the dispatch.</summary>
    [Benchmark(Description = "POH storage, direct")]
    public void Pinned()
    {
        fixed (double* pa = _ma!.View.Buffer)
        fixed (double* pb = _mb!.View.Buffer)
        fixed (double* pc = _mc!.View.Buffer)
        {
            BenchmarkKernel.Multiply(_dispatch!, N, N, N, pa, N, pb, N, pc, N);
        }
    }

    /// <summary>The shipped path: bound views through Workspace.Multiply.</summary>
    [Benchmark(Description = "public API")]
    public void PublicApi() =>
        _workspace!.Multiply(_ma!.ReadOnlyView, _mb!.ReadOnlyView, _mc!.View);

    private Matrix<double> Fill(int seed)
    {
        var matrix = new Matrix<double>(N, N);
        var rng = new Random(seed);

        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                matrix[i, j] = rng.NextDouble() - 0.5;

        return matrix;
    }

    /// <summary>Release the operands, the packing buffers and the workspace.</summary>
    public void Dispose()
    {
        if (_a is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_a); _a = null; }
        if (_b is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_b); _b = null; }
        if (_c is not null) { System.Runtime.InteropServices.NativeMemory.AlignedFree(_c); _c = null; }

        // The matrices are GC-owned; dropping the references is the cleanup.
        _ma = null;
        _mb = null;
        _mc = null;

        _dispatch?.Dispose();
        _dispatch = null;

        _workspace?.Dispose();
        _workspace = null;

        GC.SuppressFinalize(this);
    }
}
