using System.Diagnostics;
using System.Runtime.InteropServices;

namespace GemmLab;

/// <summary>
/// Verification and benchmarks for <see cref="Lu"/>.
///
/// LU is checked by residual, not by comparison against another library's
/// factors. Element-wise comparison would be wrong: on ties the pivot choice is
/// arbitrary, so two correct implementations can produce different L, U and P
/// from the same input. The residuals below are the accepted criteria.
/// </summary>
public static unsafe class LuTest
{
    private static readonly int[] BenchSizes = { 128, 256, 512, 1024, 2048 };
    private static readonly int[] BlockSizes = { 32, 64, 128, 192 };

    private const double Epsilon = 2.220446049250313e-16;

    public static void Run<TKernel>(bool verifyOnly) where TKernel : struct, IMicroKernel
    {
        Console.WriteLine();
        Console.WriteLine($"=== LU (partial pivoting, GEMM update via {TKernel.Name}) ===");

        if (!TKernel.IsSupported)
        {
            Console.WriteLine("  not supported on this CPU, skipping");
            return;
        }

        if (!Verify<TKernel>() || verifyOnly) return;

        Benchmark<TKernel>();
    }

    private static bool Verify<TKernel>() where TKernel : struct, IMicroKernel
    {
        (int rows, int columns)[] shapes =
        {
            (1, 1), (5, 5), (17, 17), (64, 64), (65, 65), (100, 100), (257, 257),
            (200, 73), (73, 200), (129, 64), (64, 129),
        };

        bool passed = true;
        double worstFactor = 0.0;
        double worstSolve = 0.0;

        using var scratch = GemmScratch.For<TKernel>();

        foreach ((int rows, int columns) in shapes)
        foreach (int blockSize in new[] { 8, 64 })
        foreach (int padding in new[] { 0, 3 })
        foreach (bool illConditioned in new[] { false, true })
        {
            int stride = rows + padding;
            double* original = RandomMatrix(rows, columns, stride, seed: 17, illConditioned);
            double* factors = CopyMatrix(original, stride * columns);

            using var lu = Lu.Factor<TKernel>(rows, columns, factors, stride, scratch, blockSize);

            double residual = FactorizationResidual<TKernel>(lu, original, stride, scratch);
            worstFactor = Math.Max(worstFactor, residual);

            double threshold = 64.0 * Math.Max(rows, columns) * Epsilon;
            if (!double.IsFinite(residual) || residual > threshold)
            {
                passed = false;
                Console.WriteLine(
                    $"  factor FAIL   : {rows}x{columns} nb={blockSize} pad={padding} "
                  + $"ill={illConditioned} residual={residual:E3} threshold={threshold:E3}");
            }

            if (rows == columns && !lu.IsSingular)
            {
                double solveResidual = SolveResidual<TKernel>(lu, original, stride, nrhs: 3);
                worstSolve = Math.Max(worstSolve, solveResidual);

                if (!double.IsFinite(solveResidual) || solveResidual > threshold)
                {
                    passed = false;
                    Console.WriteLine(
                        $"  solve FAIL    : {rows}x{columns} nb={blockSize} "
                      + $"residual={solveResidual:E3} threshold={threshold:E3}");
                }
            }

            NativeMemory.AlignedFree(original);
            NativeMemory.AlignedFree(factors);
        }

        passed &= VerifySingularDetection<TKernel>(scratch);

        Console.WriteLine($"  correctness   : {(passed ? "PASS" : "FAIL")}");
        Console.WriteLine($"  worst ||PA-LU||_F / ||A||_F : {worstFactor:E3}");
        Console.WriteLine($"  worst ||Ax-b||_inf / (||A||_inf ||x||_inf) : {worstSolve:E3}");

        if (!passed) Environment.ExitCode = 1;
        return passed;
    }

    /// <summary>
    /// Two distinct failure modes, which LAPACK also distinguishes:
    ///
    /// An exactly zero column gives an exactly zero pivot and must set
    /// SingularColumn. A duplicated column is mathematically singular but its
    /// pivot is rounding noise rather than exact zero, so the factorization
    /// completes -- the same behaviour as dgetrf. Detecting that case needs a
    /// condition estimate, and PivotRatio is the cheap stand-in.
    /// </summary>
    private static bool VerifySingularDetection<TKernel>(GemmScratch scratch)
        where TKernel : struct, IMicroKernel
    {
        const int n = 40;
        bool passed = true;

        double* exact = RandomMatrix(n, n, n, seed: 5, illConditioned: false);
        new Span<double>(exact + 7 * n, n).Clear();

        using (var lu = Lu.Factor<TKernel>(n, n, exact, n, scratch, 8))
        {
            if (!lu.IsSingular)
            {
                passed = false;
                Console.WriteLine("  singular FAIL : zero column was not reported");
            }
        }

        NativeMemory.AlignedFree(exact);

        double* duplicate = RandomMatrix(n, n, n, seed: 5, illConditioned: false);
        Buffer.MemoryCopy(duplicate + 3 * n, duplicate + 7 * n,
            (long)n * sizeof(double), (long)n * sizeof(double));

        using (var lu = Lu.Factor<TKernel>(n, n, duplicate, n, scratch, 8))
        {
            Console.WriteLine($"  rank-deficient: exact zero pivot={lu.IsSingular}, "
                            + $"pivot ratio={lu.PivotRatio:E3}");

            if (lu.PivotRatio > 1e-12)
            {
                passed = false;
                Console.WriteLine("  singular FAIL : duplicate column did not collapse a pivot");
            }
        }

        NativeMemory.AlignedFree(duplicate);
        return passed;
    }

    /// <summary>||P*A - L*U||_F / ||A||_F, computed by rebuilding the product.</summary>
    private static double FactorizationResidual<TKernel>(
        LuFactorization lu, double* original, int stride, GemmScratch scratch)
        where TKernel : struct, IMicroKernel
    {
        int m = lu.Rows, n = lu.Columns, k = Math.Min(m, n);
        if (k == 0) return 0.0;

        double* lower = Alloc((nuint)m * (nuint)k);
        double* upper = Alloc((nuint)k * (nuint)n);
        double* product = Alloc((nuint)m * (nuint)n);

        new Span<double>(lower, m * k).Clear();
        new Span<double>(upper, k * n).Clear();

        for (int j = 0; j < k; j++)
            for (int i = 0; i < m; i++)
            {
                double value = lu.Factors[(nint)j * stride + i];
                if (i > j) lower[(nint)j * m + i] = value;
                else if (i == j) lower[(nint)j * m + i] = 1.0;
            }

        for (int j = 0; j < n; j++)
            for (int i = 0; i < k && i <= j; i++)
                upper[(nint)j * k + i] = lu.Factors[(nint)j * stride + i];

        Gemm.Multiply<TKernel>(m, n, k, 1.0, lower, m, upper, k, 0.0, product, m, scratch);

        // product currently holds P*A; undo the permutation to compare with A.
        Lu.UnswapRows(product, m, 0, n, lu.Pivots, 0, k);

        double difference = 0.0, norm = 0.0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
            {
                double expected = original[(nint)j * stride + i];
                double delta = product[(nint)j * m + i] - expected;
                difference += delta * delta;
                norm += expected * expected;
            }

        NativeMemory.AlignedFree(lower);
        NativeMemory.AlignedFree(upper);
        NativeMemory.AlignedFree(product);

        return norm == 0.0 ? Math.Sqrt(difference) : Math.Sqrt(difference / norm);
    }

    /// <summary>||A*x - b||_inf / (||A||_inf * ||x||_inf), the normwise backward error.</summary>
    private static double SolveResidual<TKernel>(
        LuFactorization lu, double* original, int stride, int nrhs)
        where TKernel : struct, IMicroKernel
    {
        int n = lu.Rows;

        double* rhs = RandomMatrix(n, nrhs, n, seed: 91, illConditioned: false);
        double* solution = CopyMatrix(rhs, n * nrhs);

        Lu.Solve(lu, nrhs, solution, n);

        double matrixNorm = 0.0;
        for (int i = 0; i < n; i++)
        {
            double rowSum = 0.0;
            for (int j = 0; j < n; j++) rowSum += Math.Abs(original[(nint)j * stride + i]);
            matrixNorm = Math.Max(matrixNorm, rowSum);
        }

        double worst = 0.0;

        for (int column = 0; column < nrhs; column++)
        {
            double solutionNorm = 0.0;
            for (int i = 0; i < n; i++)
                solutionNorm = Math.Max(solutionNorm, Math.Abs(solution[(nint)column * n + i]));

            double residualNorm = 0.0;
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                    sum += original[(nint)j * stride + i] * solution[(nint)column * n + j];
                residualNorm = Math.Max(residualNorm, Math.Abs(sum - rhs[(nint)column * n + i]));
            }

            double denominator = matrixNorm * solutionNorm;
            if (denominator > 0.0) worst = Math.Max(worst, residualNorm / denominator);
        }

        NativeMemory.AlignedFree(rhs);
        NativeMemory.AlignedFree(solution);
        return worst;
    }

    private static void Benchmark<TKernel>() where TKernel : struct, IMicroKernel
    {
        using var scratch = GemmScratch.For<TKernel>();

        Console.WriteLine("     size       nb    median (ms)      GFLOP/s   % of GEMM");

        foreach (int size in BenchSizes)
        {
            double gemmGflops = MeasureGemm<TKernel>(size, scratch);

            foreach (int blockSize in BlockSizes)
            {
                if (blockSize >= size) continue;

                double* master = RandomMatrix(size, size, size, seed: 11, illConditioned: false);
                double* work = Alloc((nuint)size * (nuint)size);
                long bytes = (long)size * size * sizeof(double);

                int reps = size <= 512 ? 9 : 5;
                var samples = new double[reps];

                for (int r = -2; r < reps; r++)
                {
                    Buffer.MemoryCopy(master, work, bytes, bytes);
                    long started = Stopwatch.GetTimestamp();
                    using (Lu.Factor<TKernel>(size, size, work, size, scratch, blockSize)) { }
                    double elapsed = Stopwatch.GetElapsedTime(started).TotalSeconds;
                    if (r >= 0) samples[r] = elapsed;
                }

                Array.Sort(samples);
                double median = samples[reps / 2];

                double flops = 2.0 / 3.0 * size * size * size
                             - 0.5 * size * size
                             - 1.0 / 6.0 * size;
                double gflops = flops / median / 1e9;

                Console.WriteLine(
                    $"  {size,5}   {blockSize,6}   {median * 1e3,12:F3}   {gflops,10:F2}   "
                  + $"{100.0 * gflops / gemmGflops,10:F1}");

                NativeMemory.AlignedFree(master);
                NativeMemory.AlignedFree(work);
            }
        }
    }

    /// <summary>Same-size GEMM rate, so LU can be reported as a fraction of it.</summary>
    private static double MeasureGemm<TKernel>(int size, GemmScratch scratch)
        where TKernel : struct, IMicroKernel
    {
        double* a = RandomMatrix(size, size, size, seed: 1, illConditioned: false);
        double* b = RandomMatrix(size, size, size, seed: 2, illConditioned: false);
        double* c = Alloc((nuint)size * (nuint)size);

        for (int w = 0; w < 2; w++)
            Gemm.Multiply<TKernel>(size, size, size, 1.0, a, size, b, size, 0.0, c, size, scratch);

        int reps = 5;
        var samples = new double[reps];
        for (int r = 0; r < reps; r++)
        {
            long started = Stopwatch.GetTimestamp();
            Gemm.Multiply<TKernel>(size, size, size, 1.0, a, size, b, size, 0.0, c, size, scratch);
            samples[r] = Stopwatch.GetElapsedTime(started).TotalSeconds;
        }

        Array.Sort(samples);

        NativeMemory.AlignedFree(a);
        NativeMemory.AlignedFree(b);
        NativeMemory.AlignedFree(c);

        return 2.0 * size * size * size / samples[reps / 2] / 1e9;
    }

    /// <summary>
    /// Random matrix. The ill-conditioned variant scales columns by widely
    /// varying powers of two, which stresses pivoting without the exactness
    /// tricks a contrived test matrix would use.
    /// </summary>
    private static double* RandomMatrix(int rows, int columns, int stride, int seed, bool illConditioned)
    {
        double* p = Alloc((nuint)stride * (nuint)Math.Max(1, columns));
        var rng = new Random(seed);

        for (int j = 0; j < columns; j++)
        {
            double scale = illConditioned ? Math.Pow(2.0, rng.Next(-30, 31)) : 1.0;
            for (int i = 0; i < stride; i++)
                p[(nint)j * stride + i] = (rng.NextDouble() - 0.5) * (i < rows ? scale : 1.0);
        }

        return p;
    }

    private static double* CopyMatrix(double* source, int count)
    {
        double* p = Alloc((nuint)count);
        Buffer.MemoryCopy(source, p, (long)count * sizeof(double), (long)count * sizeof(double));
        return p;
    }

    private static double* Alloc(nuint count) =>
        (double*)NativeMemory.AlignedAlloc(count * sizeof(double), 64);
}
