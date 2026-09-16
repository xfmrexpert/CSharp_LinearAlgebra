using Tensile.Primitives;

namespace Tensile.Tests;

/// <summary>
/// The public surface, exercised the way a caller would use it.
///
/// These are as much a design check as a correctness check: anything awkward to
/// write here is awkward for a user, and the point of the ergonomic layer is
/// that the pointer-and-stride arithmetic below it never surfaces.
/// </summary>
public unsafe class MatrixApiTests
{
    private const double Tolerance = 1e-12;

    // ---- storage and views -------------------------------------------------

    [Fact]
    public void ZerosStartsZeroed()
    {
        using var a = Matrix.Zeros<double>(4, 3);

        Assert.Equal(4, a.Rows);
        Assert.Equal(3, a.Columns);
        Assert.False(a.IsSquare);
        Assert.All(a.ToArray(), value => Assert.Equal(0.0, value));
    }

    [Fact]
    public void IdentityHasUnitDiagonal()
    {
        using var a = Matrix.Identity<double>(5);

        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 5; i++)
                Assert.Equal(i == j ? 1.0 : 0.0, a[i, j]);
    }

    /// <summary>The generic storage must work for a type the operations do not yet cover.</summary>
    [Fact]
    public void StorageIsGenericEvenWhereArithmeticIsNot()
    {
        using var a = Matrix.Identity<float>(3);

        Assert.Equal(1.0f, a[2, 2]);
        Assert.Equal(0.0f, a[0, 1]);
    }

    [Fact]
    public void FromRowsReadsTheWayItIsWritten()
    {
        using var a = Matrix.FromRows(new[,]
        {
            { 1.0, 2.0, 3.0 },
            { 4.0, 5.0, 6.0 },
        });

        Assert.Equal(2, a.Rows);
        Assert.Equal(3, a.Columns);
        Assert.Equal(2.0, a[0, 1]);
        Assert.Equal(4.0, a[1, 0]);

        // Stored column-major, so ToArray walks down columns.
        Assert.Equal(new[] { 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 }, a.ToArray());
    }

    [Fact]
    public void FromColumnMajorRoundTrips()
    {
        double[] values = [1, 2, 3, 4, 5, 6];

        using var a = Matrix.FromColumnMajor<double>(3, 2, values);

        Assert.Equal(values, a.ToArray());
        Assert.Equal(4.0, a[0, 1]);
    }

    [Fact]
    public void FromColumnMajorRejectsAShapeMismatch() =>
        Assert.Throws<ArgumentException>(() => Matrix.FromColumnMajor<double>(3, 2, new double[5]));

    [Fact]
    public void ColumnsAreContiguousAndAliasTheMatrix()
    {
        using var a = Matrix.Zeros<double>(4, 3);

        a.Column(1).Fill(7.0);

        Assert.Equal(7.0, a[0, 1]);
        Assert.Equal(7.0, a[3, 1]);
        Assert.Equal(0.0, a[0, 0]);
    }

    [Fact]
    public void SliceSharesStorage()
    {
        using var a = Matrix.Zeros<double>(5, 5);

        MatrixView<double> block = a.Slice(1, 1, 2, 2);
        block.Fill(3.0);

        Assert.Equal(3.0, a[1, 1]);
        Assert.Equal(3.0, a[2, 2]);
        Assert.Equal(0.0, a[0, 0]);
        Assert.Equal(0.0, a[3, 3]);
    }

    /// <summary>A slice of a padded matrix must keep the parent's stride, not repack.</summary>
    [Fact]
    public void SliceOfAPaddedMatrixKeepsTheStride()
    {
        using var a = new Matrix<double>(4, 4, stride: 9);

        MatrixView<double> block = a.Slice(1, 1, 2, 2);

        Assert.Equal(9, block.Stride);
        Assert.False(block.IsContiguous);

        block[0, 0] = 5.0;
        Assert.Equal(5.0, a[1, 1]);
    }

    [Fact]
    public void SliceRejectsABlockThatLeavesTheMatrix()
    {
        using var a = Matrix.Zeros<double>(4, 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(2, 2, 3, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(-1, 0, 1, 1));
    }

    [Fact]
    public void IndexerRejectsOutOfRange()
    {
        using var a = Matrix.Zeros<double>(2, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => a[2, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => a[0, -1]);
    }

    [Fact]
    public void CloneIsIndependentAndPacked()
    {
        using var a = new Matrix<double>(3, 2, stride: 7);
        a[0, 0] = 1.0;

        using Matrix<double> copy = a.Clone();
        copy[0, 0] = 2.0;

        Assert.Equal(1.0, a[0, 0]);
        Assert.Equal(3, copy.Stride);
    }

    [Fact]
    public void UseAfterDisposeThrows()
    {
        var a = Matrix.Zeros<double>(2, 2);
        a.Dispose();

        // A ref struct cannot be a lambda's return value, so touch a member of it.
        Assert.Throws<ObjectDisposedException>(() => { _ = a.View.Rows; });
    }

    // ---- products ----------------------------------------------------------

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(5, 3, 4)]
    [InlineData(17, 19, 23)]
    [InlineData(64, 64, 64)]
    [InlineData(100, 37, 129)]
    public void MultiplyMatchesTheReferenceImplementation(int m, int n, int k)
    {
        using Matrix<double> a = RandomMatrix(m, k, seed: 1);
        using Matrix<double> b = RandomMatrix(k, n, seed: 2);

        using Matrix<double> product = a.Multiply(b);
        using var expected = new Matrix<double>(m, n);

        Reference.Multiply(m, n, k, 1.0,
            a.View.Pointer, a.Stride, b.View.Pointer, b.Stride,
            0.0, expected.View.Pointer, expected.Stride);

        Assert.True(MaxDifference(product, expected) < Tolerance);
    }

    [Fact]
    public void MultiplyRejectsNonConformableShapes()
    {
        using Matrix<double> a = RandomMatrix(3, 4, seed: 3);
        using Matrix<double> b = RandomMatrix(5, 2, seed: 4);

        Assert.Throws<ArgumentException>(() => a.Multiply(b));
    }

    [Fact]
    public void MultiplyIntoAccumulatesWithBeta()
    {
        using Matrix<double> a = Matrix.Identity<double>(4);
        using Matrix<double> b = RandomMatrix(4, 4, seed: 5);
        using Matrix<double> destination = b.Clone();

        // destination := 1*destination + 1*I*b  =>  2b
        a.MultiplyInto(b, destination.View, alpha: 1.0, beta: 1.0);

        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 4; i++)
                Assert.Equal(2.0 * b[i, j], destination[i, j], 12);
    }

    // ---- factorization and solving ----------------------------------------

    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(64)]
    [InlineData(129)]
    public void SolveRecoversAKnownSolution(int n)
    {
        using Matrix<double> a = RandomDiagonallyDominant(n, seed: 11);
        using Matrix<double> expected = RandomMatrix(n, 2, seed: 12);
        using Matrix<double> b = a.Multiply(expected);

        using Matrix<double> x = a.Solve(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    /// <summary>Factoring must leave the caller's matrix alone.</summary>
    [Fact]
    public void FactorLuDoesNotModifyTheInput()
    {
        using Matrix<double> a = RandomDiagonallyDominant(16, seed: 13);
        using Matrix<double> before = a.Clone();

        using LuDecomposition lu = a.FactorLu();

        Assert.Equal(0.0, MaxDifference(a, before));
    }

    [Fact]
    public void FactorizationCanBeReusedAcrossRightHandSides()
    {
        const int n = 32;

        using Matrix<double> a = RandomDiagonallyDominant(n, seed: 14);
        using LuDecomposition lu = a.FactorLu();

        for (int trial = 0; trial < 3; trial++)
        {
            using Matrix<double> expected = RandomMatrix(n, 1, seed: 100 + trial);
            using Matrix<double> b = a.Multiply(expected);
            using Matrix<double> x = lu.Solve(b);

            Assert.True(MaxDifference(x, expected) < 1e-9);
        }
    }

    [Fact]
    public void SolveTransposedRecoversAKnownSolution()
    {
        const int n = 24;

        using Matrix<double> a = RandomDiagonallyDominant(n, seed: 15);
        using LuDecomposition lu = a.FactorLu();

        using Matrix<double> expected = RandomMatrix(n, 2, seed: 16);

        // b := A^T * expected, built by multiplying with an explicit transpose.
        using Matrix<double> transpose = Transpose(a);
        using Matrix<double> b = transpose.Multiply(expected);

        using Matrix<double> x = lu.SolveTransposed(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    [Fact]
    public void SolveRejectsANonSquareMatrix()
    {
        using Matrix<double> a = RandomMatrix(4, 3, seed: 17);
        using Matrix<double> b = RandomMatrix(4, 1, seed: 18);

        Assert.Throws<ArgumentException>(() => a.Solve(b));
    }

    [Fact]
    public void DeterminantMatchesAHandComputation()
    {
        // [[1,2],[3,4]] has determinant -2.
        using var a = Matrix.FromRows(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        using LuDecomposition lu = a.FactorLu();

        Assert.Equal(-2.0, lu.Determinant(), 12);
    }

    [Fact]
    public void DeterminantOfIdentityIsOne()
    {
        using Matrix<double> a = Matrix.Identity<double>(9);
        using LuDecomposition lu = a.FactorLu();

        Assert.Equal(1.0, lu.Determinant(), 12);
    }

    [Fact]
    public void ReciprocalConditionIsOneForTheIdentity()
    {
        using Matrix<double> a = Matrix.Identity<double>(32);
        using LuDecomposition lu = a.FactorLu();

        Assert.Equal(1.0, lu.ReciprocalCondition(), 12);
    }

    /// <summary>
    /// The norm of the original is captured before the factorization overwrites
    /// it, so condition estimation cannot be handed the wrong one the way
    /// dgecon can.
    /// </summary>
    [Fact]
    public void ReciprocalConditionFallsForHilbertMatrices()
    {
        double previous = double.PositiveInfinity;

        foreach (int n in new[] { 4, 6, 8, 10 })
        {
            using var a = new Matrix<double>(n, n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    a[i, j] = 1.0 / (i + j + 1);

            using LuDecomposition lu = a.FactorLu();
            double rcond = lu.ReciprocalCondition();

            Assert.True(rcond < previous, $"rcond at n={n} was {rcond:E3}, not below {previous:E3}");
            previous = rcond;
        }
    }

    // ---- structure-typed dispatch -----------------------------------------

    /// <summary>
    /// The whole point of the structure parameter: solving through the two
    /// triangular factors by substitution must reproduce the general solve,
    /// with no factorization performed by either triangular step.
    /// </summary>
    [Fact]
    public void TriangularFactorsReproduceTheGeneralSolve()
    {
        const int n = 40;

        using Matrix<double> a = RandomDiagonallyDominant(n, seed: 21);
        using Matrix<double> b = RandomMatrix(n, 2, seed: 22);

        using LuDecomposition lu = a.FactorLu();
        using Matrix<double> expected = lu.Solve(b);

        // P*b, then forward substitution through L, then back substitution
        // through U -- each dispatched by its structure type.
        using Matrix<double> x = b.Clone();
        fixed (int* pivots = lu.Pivots.ToArray())
        {
            Lu.SwapRows(x.View.Pointer, x.Stride, 0, x.Columns, pivots, 0, n);
        }

        lu.Lower.SolveInPlace(x.View);
        lu.Upper.SolveInPlace(x.View);

        Assert.True(MaxDifference(x, expected) < Tolerance);
    }

    [Fact]
    public void UpperTriangularSolveInvertsMultiplication()
    {
        const int n = 20;

        using Matrix<double> u = UpperTriangularMatrix(n, seed: 23);
        using Matrix<double> expected = RandomMatrix(n, 3, seed: 24);
        using Matrix<double> b = u.Multiply(expected);

        using Matrix<double> x = u.As<double, UpperTriangular>().Solve(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    [Fact]
    public void LowerTriangularSolveInvertsMultiplication()
    {
        const int n = 20;

        using Matrix<double> l = LowerTriangularMatrix(n, seed: 25);
        using Matrix<double> expected = RandomMatrix(n, 3, seed: 26);
        using Matrix<double> b = l.Multiply(expected);

        using Matrix<double> x = l.As<double, LowerTriangular>().Solve(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    [Fact]
    public void TransposedTriangularSolveInvertsMultiplication()
    {
        const int n = 18;

        using Matrix<double> u = UpperTriangularMatrix(n, seed: 27);
        using Matrix<double> expected = RandomMatrix(n, 2, seed: 28);

        using Matrix<double> transpose = Transpose(u);
        using Matrix<double> b = transpose.Multiply(expected);

        using Matrix<double> x = u.As<double, UpperTriangular>().SolveTransposed(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    /// <summary>
    /// A structure claim describes which triangle is read, not that the rest is
    /// zero. Packed LU storage is the case that forces the distinction.
    /// </summary>
    [Fact]
    public void PackedFactorsDoNotClaimTheirOtherTriangleIsZero()
    {
        using Matrix<double> a = RandomDiagonallyDominant(12, seed: 29);
        using LuDecomposition lu = a.FactorLu();

        Assert.False(UpperTriangular.UnreferencedPartIsZero(lu.Upper.View));
        Assert.False(UnitLowerTriangular.UnreferencedPartIsZero(lu.Lower.View));

        // ... yet solving through them is still correct, because neither reads
        // the other's triangle. Covered by TriangularFactorsReproduceTheGeneralSolve.
    }

    [Fact]
    public void AsCheckedAcceptsAGenuineTriangleAndRejectsADenseOne()
    {
        using Matrix<double> upper = UpperTriangularMatrix(6, seed: 30);
        using Matrix<double> dense = RandomMatrix(6, 6, seed: 31);

        StructuredMatrix<double, UpperTriangular> accepted = upper.AsChecked<double, UpperTriangular>();
        Assert.Equal(6, accepted.Rows);

        Assert.Throws<ArgumentException>(() => dense.AsChecked<double, UpperTriangular>());
    }

    [Fact]
    public void TriangularSolveRejectsANonSquareOperand()
    {
        using Matrix<double> a = RandomMatrix(4, 3, seed: 32);
        using Matrix<double> b = RandomMatrix(4, 1, seed: 33);

        Assert.Throws<ArgumentException>(() => a.As<double, UpperTriangular>().Solve(b));
    }

    // ---- norms -------------------------------------------------------------

    [Fact]
    public void NormsMatchHandComputations()
    {
        using var a = Matrix.FromRows(new[,]
        {
            { 1.0, -2.0 },
            { -3.0, 4.0 },
        });

        Assert.Equal(6.0, a.OneNorm(), 12);          // max column sum: |−2|+|4|
        Assert.Equal(7.0, a.InfinityNorm(), 12);     // max row sum: |−3|+|4|
        Assert.Equal(Math.Sqrt(30.0), a.FrobeniusNorm(), 12);
    }

    [Fact]
    public void EstimatedOneNormNeverExceedsTheExactOne()
    {
        for (int n = 4; n <= 48; n += 11)
        {
            using Matrix<double> a = RandomMatrix(n, n, seed: n);

            double exact = a.OneNorm();
            double estimate = a.EstimateOneNorm();

            Assert.True(estimate <= exact * (1.0 + 1e-12), $"n={n}: {estimate:E17} > {exact:E17}");
        }
    }

    [Fact]
    public void EstimateOneNormRejectsANonSquareMatrix()
    {
        using Matrix<double> a = RandomMatrix(4, 3, seed: 34);

        Assert.Throws<ArgumentException>(() => a.EstimateOneNorm());
    }

    // ---- workspace ---------------------------------------------------------

    [Fact]
    public void AnExplicitWorkspaceAgreesWithTheSharedOne()
    {
        using Matrix<double> a = RandomMatrix(48, 48, seed: 41);
        using Matrix<double> b = RandomMatrix(48, 48, seed: 42);

        using var workspace = new Workspace(multithreaded: false);

        using Matrix<double> viaShared = a.Multiply(b);
        using Matrix<double> viaOwn = a.Multiply(b, workspace);

        Assert.True(MaxDifference(viaShared, viaOwn) < Tolerance);
        Assert.False(workspace.IsMultithreaded);
        Assert.NotEmpty(workspace.KernelName);
    }

    [Fact]
    public void DisposedWorkspaceThrows()
    {
        var workspace = new Workspace();
        workspace.Dispose();

        using Matrix<double> a = Matrix.Identity<double>(2);
        using Matrix<double> b = Matrix.Identity<double>(2);

        Assert.Throws<ObjectDisposedException>(() => a.Multiply(b, workspace));
    }

    // ---- helpers -----------------------------------------------------------

    private static Matrix<double> RandomMatrix(int rows, int columns, int seed)
    {
        var rng = new Random(seed);
        var matrix = new Matrix<double>(rows, columns);

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                matrix[i, j] = rng.NextDouble() - 0.5;

        return matrix;
    }

    private static Matrix<double> RandomDiagonallyDominant(int n, int seed)
    {
        Matrix<double> matrix = RandomMatrix(n, n, seed);
        for (int i = 0; i < n; i++) matrix[i, i] += n;
        return matrix;
    }

    private static Matrix<double> UpperTriangularMatrix(int n, int seed)
    {
        var rng = new Random(seed);
        var matrix = new Matrix<double>(n, n);

        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < j; i++) matrix[i, j] = rng.NextDouble() - 0.5;
            matrix[j, j] = 1.0 + rng.NextDouble();
        }

        return matrix;
    }

    private static Matrix<double> LowerTriangularMatrix(int n, int seed)
    {
        var rng = new Random(seed);
        var matrix = new Matrix<double>(n, n);

        for (int j = 0; j < n; j++)
        {
            matrix[j, j] = 1.0 + rng.NextDouble();
            for (int i = j + 1; i < n; i++) matrix[i, j] = rng.NextDouble() - 0.5;
        }

        return matrix;
    }

    private static Matrix<double> Transpose(Matrix<double> a)
    {
        var result = new Matrix<double>(a.Columns, a.Rows);

        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < a.Rows; i++)
                result[j, i] = a[i, j];

        return result;
    }

    private static double MaxDifference(Matrix<double> x, Matrix<double> y)
    {
        double worst = 0.0;

        for (int j = 0; j < x.Columns; j++)
            for (int i = 0; i < x.Rows; i++)
                worst = Math.Max(worst, Math.Abs(x[i, j] - y[i, j]));

        return worst;
    }
}
