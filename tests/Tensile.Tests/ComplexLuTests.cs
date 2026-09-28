using System.Numerics;

namespace Tensile.Tests;

/// <summary>
/// Complex LU, per micro-kernel, and the generic algorithm behind it checked
/// against the real kernel LU.
///
/// The complex factorization is verified by residual, like the real one: on a
/// tie the pivot choice is arbitrary, so no element-wise comparison against
/// another implementation is meaningful. The generic algorithm, though, can be
/// held to something much stronger. Instantiated for double it applies the
/// same pivot rule with the same arithmetic as the kernel LU, so it must pick
/// exactly the same pivots, and its factors must agree to rounding. That
/// comparison is the oracle for the code the complex path shares with it.
/// </summary>
public abstract class ComplexLuContract<TCase> where TCase : struct, IKernelCase
{
    internal static readonly KernelDriver Kernel = KernelDriver.For<TCase>();

    protected ComplexLuContract() =>
        Assert.SkipUnless(Kernel.IsSupported, $"{Kernel.Name} is not supported on this CPU");

    /// <summary>Square, tall and wide, either side of the block size, with and without stride padding.</summary>
    public static TheoryData<int, int, int, int> Cases
    {
        get
        {
            var data = new TheoryData<int, int, int, int>();

            (int rows, int columns)[] shapes =
            [
                (1, 1), (2, 2), (5, 5), (17, 17), (33, 33), (65, 65), (100, 100), (129, 129), (200, 73), (73, 200),
            ];

            foreach ((int rows, int columns) in shapes)
                foreach (int blockSize in new[] { 8, 32 })
                    foreach (int padding in new[] { 0, 3 })
                        data.Add(rows, columns, blockSize, padding);

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FactorizationResidualIsSmall(int rows, int columns, int blockSize, int padding)
    {
        using var workspace = Kernel.Workspace(multithreaded: (rows + blockSize) % 2 == 0);

        var original = ComplexLuTests.Random(rows, columns, seed: (rows * 31) + columns, stride: rows + padding);
        LuDecomposition<Complex> lu = workspace.FactorLu(original.Clone(), blockSize);

        double residual = ComplexLuTests.FactorizationResidual(lu, original);

        Assert.True(residual < 1e-12, $"||PA-LU||/||A|| = {residual:E3}");
    }

    /// <summary>
    /// The generic algorithm for double against the kernel LU: the same pivot
    /// rule on the same arithmetic, so the same pivots, exactly, and the same
    /// factors to rounding. The trailing updates are the same GEMM in both.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void GenericDoubleMatchesTheKernelFactorization(int rows, int columns, int blockSize, int padding)
    {
        using var workspace = Kernel.Workspace(multithreaded: false);

        var original = ComplexLuTests.RandomReal(rows, columns, seed: (rows * 17) + columns, stride: rows + padding);

        LuDecomposition<double> kernel = workspace.FactorLu(original.Clone(), blockSize);
        LuDecomposition<double> generic = workspace.FactorLuManaged<double, DoubleKernels>(original.Clone(), blockSize);

        Assert.Equal(kernel.Pivots.ToArray(), generic.Pivots.ToArray());
        Assert.Equal(kernel.SingularColumn, generic.SingularColumn);

        double scale = ComplexLuTests.MaxAbs(kernel.Upper.View);
        double difference = ComplexLuTests.MaxDifference(kernel.Upper.View, generic.Upper.View);
        Assert.True(difference <= 1e-13 * Math.Max(scale, 1.0), $"factors differ by {difference:E3}");
    }

    /// <summary>
    /// A real matrix factored as complex: |Re| + |Im| is |x| on the real
    /// axis, and 4M with zero imaginary parts is the real product, so the
    /// complex path must pivot exactly as the real kernel does and leave the
    /// imaginary parts zero.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void ARealMatrixPivotsAsTheRealFactorizationDoes(int rows, int columns, int blockSize, int padding)
    {
        using var workspace = Kernel.Workspace(multithreaded: false);

        var real = ComplexLuTests.RandomReal(rows, columns, seed: (rows * 13) + columns, stride: rows + padding);
        var complex = new Matrix<Complex>(rows, columns);
        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                complex[i, j] = real[i, j];

        LuDecomposition<double> expected = workspace.FactorLu(real.Clone(), blockSize);
        LuDecomposition<Complex> actual = workspace.FactorLu(complex, blockSize);

        Assert.Equal(expected.Pivots.ToArray(), actual.Pivots.ToArray());

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                Assert.Equal(0.0, actual.Upper.View[i, j].Imaginary);
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(7, 3)]
    [InlineData(50, 4)]
    [InlineData(129, 2)]
    public void SolvesHaveSmallBackwardError(int n, int rightHandSides)
    {
        using var workspace = Kernel.Workspace(multithreaded: false);

        var a = ComplexLuTests.Random(n, n, seed: n + 3);
        var b = ComplexLuTests.Random(n, rightHandSides, seed: n + 4);
        LuDecomposition<Complex> lu = workspace.FactorLu(a.Clone(), blockSize: 16);

        Matrix<Complex> x = lu.Solve(b.ReadOnlyView);
        Assert.True(
            ComplexLuTests.BackwardError(a, x, b) < 1e-14,
            $"A x = b backward error {ComplexLuTests.BackwardError(a, x, b):E3}");

        Matrix<Complex> adjoint = ComplexLuTests.Adjoint(a);
        Matrix<Complex> y = lu.SolveAdjoint(b.ReadOnlyView);
        Assert.True(
            ComplexLuTests.BackwardError(adjoint, y, b) < 1e-14,
            $"A^H y = b backward error {ComplexLuTests.BackwardError(adjoint, y, b):E3}");
    }
}

public sealed class ScalarComplexLuTests : ComplexLuContract<ScalarCase>;
public sealed class Avx2ComplexLuTests : ComplexLuContract<Avx2Case>;
public sealed class Avx512ComplexLuTests : ComplexLuContract<Avx512Case>;

/// <summary>The complex LU's surface and the properties that do not depend on the micro-kernel.</summary>
public class ComplexLuTests
{
    // ---- the pivot rule ------------------------------------------------------

    /// <summary>
    /// 3 has the larger modulus (3 against 2.83) and 2+2i the larger
    /// |Re| + |Im| (4 against 3). LAPACK's izamax picks by the second, and so
    /// must this.
    /// </summary>
    [Fact]
    public void PivotingMaximisesTheSumOfAbsoluteParts()
    {
        var a = Matrix.FromRows(new Complex[,]
        {
            { 3.0, 1.0 },
            { new Complex(2.0, 2.0), 1.0 },
        });

        Assert.Equal(1, a.FactorLu().Pivots[0]);
    }

    /// <summary>Equal measures resolve to the lowest index, as idamax does, whatever the moduli.</summary>
    [Fact]
    public void PivotTiesGoToTheLowestIndex()
    {
        var a = Matrix.FromRows(new Complex[,]
        {
            { new Complex(1.0, 1.0), 1.0 },
            { new Complex(-2.0, 0.0), 2.0 },
            { new Complex(0.5, -1.5), 3.0 },
        });

        Assert.Equal(0, a.FactorLu().Pivots[0]);
    }

    [Fact]
    public void ANaNNeverWinsThePivotSearch()
    {
        ReadOnlySpan<Complex> column = [new Complex(1.0, 0.0), new Complex(double.NaN, 0.0), new Complex(0.0, 2.0)];

        Assert.Equal(2, BlockedLu.IndexOfLargestPivot<Complex, ComplexKernels>(column));
    }

    // ---- solves --------------------------------------------------------------

    /// <summary>
    /// The embedding is an independent route to the same solution: a real LU
    /// of [[X, -Y], [Y, X]] on the stacked right-hand side. Different pivots,
    /// different arithmetic, same answer.
    /// </summary>
    [Fact]
    public void SolveAgreesWithTheRealEmbedding()
    {
        const int N = 40;

        var a = DiagonallyDominant(N, seed: 5);
        var b = Random(N, 3, seed: 6);

        Matrix<Complex> x = a.Solve(b.ReadOnlyView);
        Matrix<Complex> viaEmbedding = ComplexEmbedding.Unstack(
            ComplexEmbedding.Embed(a).Solve(ComplexEmbedding.Stack(b.ReadOnlyView).ReadOnlyView));

        Assert.True(MaxDifference(x.ReadOnlyView, viaEmbedding.ReadOnlyView) < 1e-12 * MaxAbs(x.ReadOnlyView));
    }

    [Fact]
    public void TheInverseOperatorAppliesTheSolves()
    {
        const int N = 12;

        var a = DiagonallyDominant(N, seed: 7);
        var x = Random(N, 2, seed: 8);
        var inverse = new LuInverseOperator<Complex>(a.FactorLu());

        var y = new Matrix<Complex>(N, 2);
        inverse.Apply(a.Multiply(x.ReadOnlyView).ReadOnlyView, y.View);
        Assert.True(MaxDifference(x.ReadOnlyView, y.ReadOnlyView) < 1e-12);

        inverse.ApplyAdjoint(Adjoint(a).Multiply(x.ReadOnlyView).ReadOnlyView, y.View);
        Assert.True(MaxDifference(x.ReadOnlyView, y.ReadOnlyView) < 1e-12);
    }

    [Fact]
    public void AnExactlySingularMatrixIsReportedAndRefusesToSolve()
    {
        var a = Random(6, 6, seed: 9);
        for (int i = 0; i < 6; i++) a[i, 3] = Complex.Zero;

        LuDecomposition<Complex> lu = a.FactorLu(blockSize: 2);

        Assert.True(lu.IsSingular);
        Assert.Equal(3, lu.SingularColumn);
        Assert.Throws<InvalidOperationException>(() => lu.Solve(Random(6, 1, seed: 10).ReadOnlyView));
        Assert.Throws<InvalidOperationException>(() => lu.SolveAdjoint(Random(6, 1, seed: 10).ReadOnlyView));
    }

    [Fact]
    public void ANonSquareFactorizationRefusesToSolve()
    {
        LuDecomposition<Complex> lu = Random(5, 3, seed: 11).FactorLu();

        Assert.Throws<InvalidOperationException>(() => lu.Solve(Random(5, 1, seed: 12).ReadOnlyView));
        Assert.Throws<ArgumentException>(() => Random(5, 3, seed: 11).Solve(Random(5, 1, seed: 12).ReadOnlyView));
    }

    [Fact]
    public void AMismatchedRightHandSideIsRejected()
    {
        LuDecomposition<Complex> lu = DiagonallyDominant(4, seed: 13).FactorLu();

        Assert.Throws<ArgumentException>(() => lu.Solve(Random(5, 1, seed: 14).ReadOnlyView));
    }

    // ---- determinant and diagnostics ------------------------------------------

    /// <summary>[[0, 2i], [3, 0]] needs one interchange: det = -(3 * 2i) = -6i.</summary>
    [Fact]
    public void TheDeterminantCarriesThePermutationSign()
    {
        var a = Matrix.FromRows(new Complex[,]
        {
            { 0.0, new Complex(0.0, 2.0) },
            { 3.0, 0.0 },
        });

        Complex det = a.FactorLu().Determinant();

        Assert.Equal(0.0, det.Real, 14);
        Assert.Equal(-6.0, det.Imaginary, 14);
    }

    /// <summary>
    /// det([[X, -Y], [Y, X]]) = |det(X + iY)|^2, so the real LU of the
    /// embedding checks the complex determinant through a different
    /// factorization entirely.
    /// </summary>
    [Fact]
    public void TheDeterminantAgreesWithTheEmbedding()
    {
        var a = Random(6, 6, seed: 15);

        double modulusSquared = Math.Pow(Complex.Abs(a.FactorLu().Determinant()), 2);
        double embedded = ComplexEmbedding.Embed(a).FactorLu().Determinant();

        Assert.True(Math.Abs(modulusSquared - embedded) <= 1e-12 * Math.Abs(embedded));
    }

    [Fact]
    public void TheIdentityFactorsTrivially()
    {
        LuDecomposition<Complex> lu = Matrix.Identity<Complex>(9).FactorLu(blockSize: 4);

        Assert.Equal(Complex.One, lu.Determinant());
        Assert.Equal(1.0, lu.PivotRatio);
        Assert.Equal(Enumerable.Range(0, 9).ToArray(), lu.Pivots.ToArray());
    }

    // ---- the surface's promises -----------------------------------------------

    [Fact]
    public void TheFluentFactorizationLeavesItsInputAlone()
    {
        var a = Random(10, 10, seed: 16);
        var before = a.Clone();

        _ = a.FactorLu(blockSize: 3);

        Assert.Equal(0.0, MaxDifference(a.ReadOnlyView, before.ReadOnlyView));
    }

    [Fact]
    public void ADisposedWorkspaceFailsBeforeTouchingTheMatrix()
    {
        var workspace = new Workspace(multithreaded: false);
        workspace.Dispose();

        var a = Random(8, 8, seed: 17);
        var before = a.Clone();

        Assert.Throws<ObjectDisposedException>(() => workspace.FactorLu(a));
        Assert.Equal(0.0, MaxDifference(a.ReadOnlyView, before.ReadOnlyView));
    }

    /// <summary>Finding 11: an empty matrix with two billion nominal columns costs nothing to factor.</summary>
    [Fact]
    public void AnEmptyMatrixWithHugeExtentFactorsImmediately()
    {
        var a = new Matrix<Complex>(0, 1_000_000_000);

        var clock = System.Diagnostics.Stopwatch.StartNew();
        LuDecomposition<Complex> lu = Workspace.Shared.FactorLu(a);

        Assert.Equal(0, lu.Pivots.Length);
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1), $"took {clock.Elapsed}");
    }

    // ---- helpers ---------------------------------------------------------------

    internal static Matrix<Complex> Random(int rows, int columns, int seed, int stride = 0)
    {
        var m = new Matrix<Complex>(rows, columns, stride);
        var rng = new Random(seed);

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                m[i, j] = new Complex((rng.NextDouble() * 2.0) - 1.0, (rng.NextDouble() * 2.0) - 1.0);

        return m;
    }

    internal static Matrix<double> RandomReal(int rows, int columns, int seed, int stride = 0)
    {
        var m = new Matrix<double>(rows, columns, stride);
        var rng = new Random(seed);

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                m[i, j] = (rng.NextDouble() * 2.0) - 1.0;

        return m;
    }

    internal static Matrix<Complex> DiagonallyDominant(int n, int seed)
    {
        var m = Random(n, n, seed);
        for (int i = 0; i < n; i++) m[i, i] += 2.0 * n;
        return m;
    }

    internal static Matrix<Complex> Adjoint(Matrix<Complex> a)
    {
        var result = new Matrix<Complex>(a.Columns, a.Rows);
        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[j, i] = Complex.Conjugate(a[i, j]);
        return result;
    }

    /// <summary>||P*A - L*U||_F / ||A||_F, with L and U read out of the packed factors.</summary>
    internal static double FactorizationResidual(LuDecomposition<Complex> lu, Matrix<Complex> original)
    {
        int m = lu.Rows, n = lu.Columns, k = Math.Min(m, n);
        ReadOnlyMatrixView<Complex> factors = lu.Upper.View;

        var pa = Matrix.From(original.ReadOnlyView);
        ReadOnlySpan<int> pivots = lu.Pivots;
        for (int step = 0; step < pivots.Length; step++)
        {
            if (pivots[step] == step) continue;
            for (int j = 0; j < n; j++) (pa[step, j], pa[pivots[step], j]) = (pa[pivots[step], j], pa[step, j]);
        }

        double difference = 0.0, norm = 0.0;

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                Complex sum = Complex.Zero;
                int depth = Math.Min(Math.Min(i, j) + 1, k);

                for (int l = 0; l < depth; l++)
                {
                    Complex lower = i == l ? Complex.One : factors[i, l];
                    sum += lower * factors[l, j];
                }

                difference += Math.Pow(Complex.Abs(pa[i, j] - sum), 2);
                norm += Math.Pow(Complex.Abs(original[i, j]), 2);
            }
        }

        return norm == 0.0 ? Math.Sqrt(difference) : Math.Sqrt(difference / norm);
    }

    /// <summary>||A*X - B||_inf / (||A||_inf ||X||_inf + ||B||_inf), columnwise worst.</summary>
    internal static double BackwardError(Matrix<Complex> a, Matrix<Complex> x, Matrix<Complex> b)
    {
        Matrix<Complex> ax = a.Multiply(x.ReadOnlyView);
        double normA = ComplexKernels.InfinityNorm(a.ReadOnlyView);
        double worst = 0.0;

        for (int j = 0; j < x.Columns; j++)
        {
            double residual = 0.0, normX = 0.0, normB = 0.0;

            for (int i = 0; i < x.Rows; i++)
            {
                residual = Math.Max(residual, Complex.Abs(ax[i, j] - b[i, j]));
                normX = Math.Max(normX, Complex.Abs(x[i, j]));
                normB = Math.Max(normB, Complex.Abs(b[i, j]));
            }

            worst = Math.Max(worst, residual / ((normA * normX) + normB));
        }

        return worst;
    }

    internal static double MaxAbs(ReadOnlyMatrixView<double> a)
    {
        double best = 0.0;
        for (int j = 0; j < a.Columns; j++)
            foreach (double value in a.Column(j)) best = Math.Max(best, Math.Abs(value));
        return best;
    }

    internal static double MaxAbs(ReadOnlyMatrixView<Complex> a)
    {
        double best = 0.0;
        for (int j = 0; j < a.Columns; j++)
            foreach (Complex value in a.Column(j)) best = Math.Max(best, Complex.Abs(value));
        return best;
    }

    internal static double MaxDifference(ReadOnlyMatrixView<double> a, ReadOnlyMatrixView<double> b)
    {
        double best = 0.0;
        for (int j = 0; j < a.Columns; j++)
        {
            ReadOnlySpan<double> x = a.Column(j), y = b.Column(j);
            for (int i = 0; i < x.Length; i++) best = Math.Max(best, Math.Abs(x[i] - y[i]));
        }

        return best;
    }

    internal static double MaxDifference(ReadOnlyMatrixView<Complex> a, ReadOnlyMatrixView<Complex> b)
    {
        double best = 0.0;
        for (int j = 0; j < a.Columns; j++)
        {
            ReadOnlySpan<Complex> x = a.Column(j), y = b.Column(j);
            for (int i = 0; i < x.Length; i++) best = Math.Max(best, Complex.Abs(x[i] - y[i]));
        }

        return best;
    }
}
