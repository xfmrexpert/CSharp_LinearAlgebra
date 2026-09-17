using BenchmarkDotNet.Attributes;

namespace Tensile.Benchmarks;

/// <summary>
/// LU factorization, and the block-size sweep that goes with it.
///
/// The block-size optimum shifts with n, which is why it is a parameter rather
/// than a constant: on the verification container nb=32 won below n=2048 and
/// nb=64 at it. A size-dependent default is probably right once this has been
/// run on real hardware.
///
/// Read the result against <see cref="GemmBenchmarks"/> at the same size and
/// the same threading. LU is a GEMM with a serial panel bolted on, so the
/// interesting number is the ratio, and a threaded LU compared against a serial
/// GEMM measures the thread count instead.
/// </summary>
[MemoryDiagnoser(displayGenColumns: false)]
public class LuBenchmarks : IDisposable
{
    private Matrix<double>? _original;
    private Matrix<double>? _work;
    private Workspace? _workspace;

    /// <summary>Square problem size.</summary>
    [Params(256, 512, 1024, 2048)]
    public int N { get; set; }

    /// <summary>Panel width. Zero would select the library default; these are the sweep.</summary>
    [Params(32, 64, 128)]
    public int BlockSize { get; set; }

    /// <summary>Build a well-conditioned operand once per parameter set.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(17);
        _original = new Matrix<double>(N, N);

        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                _original[i, j] = rng.NextDouble() - 0.5;

        for (int i = 0; i < N; i++) _original[i, i] += N;

        _work = new Matrix<double>(N, N);
        _workspace = new Workspace(multithreaded: true);
    }

    /// <summary>Release the operands and workspace.</summary>
    [GlobalCleanup]
    public void Cleanup() => Dispose();

    /// <summary>
    /// Factor in place, restoring the operand first so every iteration starts
    /// from the same well-conditioned matrix rather than from the previous
    /// iteration's factors.
    /// </summary>
    [Benchmark]
    public void Factor()
    {
        _original!.View.CopyTo(_work!.View);

        using Tensile.Primitives.LuFactorization factorization =
            _workspace!.FactorLu(_work.View, BlockSize);
    }

    /// <summary>Release the operands and workspace.</summary>
    public void Dispose()
    {
        // The matrices are GC-owned; dropping the references is the whole
        // cleanup. Only the workspace holds native buffers.
        _original = null;
        _work = null;

        _workspace?.Dispose();
        _workspace = null;

        GC.SuppressFinalize(this);
    }
}
