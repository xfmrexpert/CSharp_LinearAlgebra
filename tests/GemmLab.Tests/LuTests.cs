namespace GemmLab.Tests;

/// <summary>
/// LU, checked by residual.
///
/// Never element-wise against another factorization: on a tie the pivot choice
/// is arbitrary, so two correct implementations legitimately produce different
/// L, U and P from the same input. What must hold is ||PA - LU|| / ||A|| near
/// machine precision, and a solve whose backward error is of the same order.
/// </summary>
public abstract unsafe class LuContract<TKernel> where TKernel : struct, IMicroKernel
{
    private const double Tolerance = 1e-12;

    protected LuContract() =>
        Assert.SkipUnless(TKernel.IsSupported, $"{TKernel.Name} is not supported on this CPU");

    /// <summary>
    /// Square, tall and wide, either side of the block size, with and without
    /// stride padding.
    /// </summary>
    public static TheoryData<int, int, int, int> Cases
    {
        get
        {
            var data = new TheoryData<int, int, int, int>();

            (int rows, int columns)[] shapes =
            [
                (1, 1), (2, 2), (5, 5), (17, 17), (32, 32), (33, 33), (64, 64), (65, 65),
                (100, 100), (129, 129), (200, 73), (73, 200), (129, 64), (64, 129),
            ];

            foreach ((int rows, int columns) in shapes)
                foreach (int blockSize in new[] { 8, 32, 64 })
                    foreach (int padding in new[] { 0, 3 })
                        data.Add(rows, columns, blockSize, padding);

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FactorizationResidualIsSmall(int rows, int columns, int blockSize, int padding)
    {
        using var original = Matrix.Random(rows, columns, seed: 42, stride: rows + padding);
        using var factors = original.Clone();
        using var gemm = GemmDispatch.Multithreaded<TKernel>();

        using var lu = Lu.Factor<TKernel>(rows, columns, factors.Data, factors.Stride, gemm, blockSize);

        double residual = FactorizationResidual(lu, original);

        Assert.True(residual < Tolerance, $"||PA-LU||/||A|| = {residual:E3}");
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(16, 8)]
    [InlineData(64, 32)]
    [InlineData(129, 64)]
    [InlineData(200, 64)]
    public void SolveHasSmallBackwardError(int n, int blockSize)
    {
        using var original = Matrix.RandomDiagonallyDominant(n, seed: 7, stride: n + 2);
        using var factors = original.Clone();
        using var b = Matrix.Random(n, 3, seed: 8, stride: n + 1);
        using var x = b.Clone();
        using var gemm = GemmDispatch.Multithreaded<TKernel>();

        using var lu = Lu.Factor<TKernel>(n, n, factors.Data, factors.Stride, gemm, blockSize);
        Lu.Solve(lu, x.Columns, x.Data, x.Stride);

        double error = ResidualOfSolve(original, x, b, transposed: false);

        Assert.True(error < Tolerance, $"||Ax-b|| relative = {error:E3}");
    }

    [Theory]
    [InlineData(1, 8)]
    [InlineData(16, 8)]
    [InlineData(64, 32)]
    [InlineData(129, 64)]
    [InlineData(200, 64)]
    public void SolveTransposedHasSmallBackwardError(int n, int blockSize)
    {
        using var original = Matrix.RandomDiagonallyDominant(n, seed: 9, stride: n + 2);
        using var factors = original.Clone();
        using var b = Matrix.Random(n, 3, seed: 10, stride: n + 1);
        using var x = b.Clone();
        using var gemm = GemmDispatch.Multithreaded<TKernel>();

        using var lu = Lu.Factor<TKernel>(n, n, factors.Data, factors.Stride, gemm, blockSize);
        Lu.SolveTransposed(lu, x.Columns, x.Data, x.Stride);

        double error = ResidualOfSolve(original, x, b, transposed: true);

        Assert.True(error < Tolerance, $"||A^T x - b|| relative = {error:E3}");
    }

    /// <summary>
    /// A block size at or above min(m,n) takes the unblocked shortcut, which
    /// must agree with the blocked path.
    ///
    /// Agree, but not bit for bit: the blocked path puts its trailing update
    /// through GEMM, which sums the same products in a different order, so the
    /// factors differ in the last couple of bits. The pivot sequence is the
    /// part that must match exactly -- it is a sequence of integer choices, and
    /// a rounding difference large enough to flip one would mean the matrix was
    /// on a knife edge rather than that either path was wrong.
    /// </summary>
    [Fact]
    public void UnblockedShortcutAgreesWithBlockedPath()
    {
        const int n = 48;

        using var original = Matrix.Random(n, n, seed: 11);
        using var gemm = GemmDispatch.Serial<TKernel>();

        using var blockedFactors = original.Clone();
        using var unblockedFactors = original.Clone();

        using var blocked = Lu.Factor<TKernel>(n, n, blockedFactors.Data, blockedFactors.Stride, gemm, 8);
        using var unblocked = Lu.Factor<TKernel>(n, n, unblockedFactors.Data, unblockedFactors.Stride, gemm, n);

        for (int i = 0; i < n; i++)
            Assert.Equal(blocked.Pivots[i], unblocked.Pivots[i]);

        double difference = blockedFactors.MaxDifference(unblockedFactors);
        Assert.True(difference < 1e-12, $"blocked vs unblocked factors differ by {difference:E3}");

        // Both must actually be factorizations of the original, not merely of
        // each other.
        Assert.True(FactorizationResidual(blocked, original) < Tolerance);
        Assert.True(FactorizationResidual(unblocked, original) < Tolerance);
    }

    /// <summary>
    /// An exactly singular matrix must be reported, and a solve against it must
    /// throw rather than return infinities.
    /// </summary>
    [Fact]
    public void ExactZeroPivotIsReported()
    {
        const int n = 16;

        using var a = Matrix.RandomDiagonallyDominant(n, seed: 12);
        for (int i = 0; i < n; i++) a[i, 5] = 0.0;

        using var gemm = GemmDispatch.Serial<TKernel>();
        using var lu = Lu.Factor<TKernel>(n, n, a.Data, a.Stride, gemm, 4);

        Assert.True(lu.IsSingular);
        Assert.Equal(5, lu.SingularColumn);

        using var b = Matrix.Random(n, 1, seed: 13);
        Assert.Throws<InvalidOperationException>(() => Lu.Solve(lu, 1, b.Data, b.Stride));
        Assert.Throws<InvalidOperationException>(() => Lu.SolveTransposed(lu, 1, b.Data, b.Stride));
    }

    /// <summary>
    /// A duplicated column is mathematically singular, but its pivot comes out
    /// as rounding noise rather than an exact zero, so the factorization
    /// completes -- identical to dgetrf. PivotRatio is the signal, and it is an
    /// indicator only.
    /// </summary>
    [Fact]
    public void DuplicatedColumnFactorsButLooksIllConditioned()
    {
        const int n = 24;

        using var a = Matrix.RandomDiagonallyDominant(n, seed: 14);
        for (int i = 0; i < n; i++) a[i, 9] = a[i, 3];

        using var gemm = GemmDispatch.Serial<TKernel>();
        using var lu = Lu.Factor<TKernel>(n, n, a.Data, a.Stride, gemm, 8);

        Assert.False(lu.IsSingular);
        Assert.True(lu.PivotRatio < 1e-14, $"pivot ratio {lu.PivotRatio:E3} should be tiny");
    }

    [Fact]
    public void NonSquareSolveIsRejected()
    {
        using var a = Matrix.Random(12, 8, seed: 15);
        using var gemm = GemmDispatch.Serial<TKernel>();
        using var lu = Lu.Factor<TKernel>(12, 8, a.Data, a.Stride, gemm, 4);

        using var b = Matrix.Random(12, 1, seed: 16);

        Assert.Throws<ArgumentException>(() => Lu.Solve(lu, 1, b.Data, b.Stride));
        Assert.Throws<ArgumentException>(() => Lu.SolveTransposed(lu, 1, b.Data, b.Stride));
    }

    /// <summary>||PA - LU||_F / ||A||_F, rebuilding PA from the packed factors.</summary>
    private static double FactorizationResidual(LuFactorization lu, Matrix original)
    {
        int m = lu.Rows;
        int n = lu.Columns;
        int k = Math.Min(m, n);

        // Reconstruct L*U into a dense product, then undo the permutation.
        using var product = new Matrix(m, n);

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                double sum = 0.0;

                for (int p = 0; p < k; p++)
                {
                    double lower = i == p ? 1.0 : (i > p ? lu.Factors[(nint)p * lu.Stride + i] : 0.0);
                    double upper = p <= j ? lu.Factors[(nint)j * lu.Stride + p] : 0.0;
                    sum += lower * upper;
                }

                product[i, j] = sum;
            }
        }

        Lu.UnswapRows(product.Data, product.Stride, 0, n, lu.Pivots, 0, k);

        double difference = 0.0;
        double norm = 0.0;

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                double value = original[i, j];
                double delta = product[i, j] - value;
                difference += delta * delta;
                norm += value * value;
            }
        }

        return norm == 0.0 ? Math.Sqrt(difference) : Math.Sqrt(difference) / Math.Sqrt(norm);
    }

    /// <summary>||A x - b||_inf / (||A||_inf ||x||_inf), or the transposed form.</summary>
    private static double ResidualOfSolve(Matrix a, Matrix x, Matrix b, bool transposed)
    {
        int n = a.Rows;
        double worst = 0.0;
        double normX = 0.0;

        for (int col = 0; col < x.Columns; col++)
            for (int i = 0; i < n; i++)
                normX = Math.Max(normX, Math.Abs(x[i, col]));

        double normA = transposed
            ? Norms.One(n, n, a.Data, a.Stride)
            : Norms.Infinity(n, n, a.Data, a.Stride);

        for (int col = 0; col < x.Columns; col++)
        {
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++) sum += (transposed ? a[j, i] : a[i, j]) * x[j, col];

                worst = Math.Max(worst, Math.Abs(sum - b[i, col]));
            }
        }

        return normA == 0.0 || normX == 0.0 ? worst : worst / (normA * normX);
    }
}

public sealed class ScalarLuTests : LuContract<ScalarKernel4x4>;
public sealed class Avx2LuTests : LuContract<Avx2Kernel8x6>;
public sealed class Avx512LuTests : LuContract<Avx512Kernel16x8>;
