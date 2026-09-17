using Tensile.Kernels;

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
        var a = Matrix.Zeros<double>(4, 3);

        Assert.Equal(4, a.Rows);
        Assert.Equal(3, a.Columns);
        Assert.False(a.IsSquare);
        Assert.All(a.ToArray(), value => Assert.Equal(0.0, value));
    }

    [Fact]
    public void IdentityHasUnitDiagonal()
    {
        var a = Matrix.Identity<double>(5);

        for (int j = 0; j < 5; j++)
            for (int i = 0; i < 5; i++)
                Assert.Equal(i == j ? 1.0 : 0.0, a[i, j]);
    }

    /// <summary>The generic storage must work for a type the operations do not yet cover.</summary>
    [Fact]
    public void StorageIsGenericEvenWhereArithmeticIsNot()
    {
        var a = Matrix.Identity<float>(3);

        Assert.Equal(1.0f, a[2, 2]);
        Assert.Equal(0.0f, a[0, 1]);
    }

    [Fact]
    public void FromRowsReadsTheWayItIsWritten()
    {
        var a = Matrix.FromRows(new[,]
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

        var a = Matrix.FromColumnMajor<double>(3, 2, values);

        Assert.Equal(values, a.ToArray());
        Assert.Equal(4.0, a[0, 1]);
    }

    [Fact]
    public void FromColumnMajorRejectsAShapeMismatch() =>
        Assert.Throws<ArgumentException>(() => Matrix.FromColumnMajor<double>(3, 2, new double[5]));

    [Fact]
    public void ColumnsAreContiguousAndAliasTheMatrix()
    {
        var a = Matrix.Zeros<double>(4, 3);

        a.Column(1).Fill(7.0);

        Assert.Equal(7.0, a[0, 1]);
        Assert.Equal(7.0, a[3, 1]);
        Assert.Equal(0.0, a[0, 0]);
    }

    [Fact]
    public void SliceSharesStorage()
    {
        var a = Matrix.Zeros<double>(5, 5);

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
        var a = new Matrix<double>(4, 4, stride: 9);

        MatrixView<double> block = a.Slice(1, 1, 2, 2);

        Assert.Equal(9, block.Stride);
        Assert.False(block.IsContiguous);

        block[0, 0] = 5.0;
        Assert.Equal(5.0, a[1, 1]);
    }

    [Fact]
    public void SliceRejectsABlockThatLeavesTheMatrix()
    {
        var a = Matrix.Zeros<double>(4, 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(2, 2, 3, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Slice(-1, 0, 1, 1));
    }

    [Fact]
    public void IndexerRejectsOutOfRange()
    {
        var a = Matrix.Zeros<double>(2, 2);

        Assert.Throws<ArgumentOutOfRangeException>(() => a[2, 0]);
        Assert.Throws<ArgumentOutOfRangeException>(() => a[0, -1]);
    }

    /// <summary>
    /// The bounds check must not be computed as row + rows: that sum overflows
    /// int for large arguments and wraps negative, which passes a "<= Rows"
    /// comparison and hands back a view pointing outside the buffer. Before the
    /// fix this returned a 1x2000000000 view about 16 GB past a 1x4 matrix.
    /// </summary>
    [Fact]
    public void SliceRejectsArgumentsThatOverflowTheBoundsCheck()
    {
        var a = Matrix.Zeros<double>(1, 4);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => a.Slice(0, 2_000_000_000, 1, 2_000_000_000));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => a.Slice(2_000_000_000, 0, 2_000_000_000, 1));

        // The same arithmetic, on the read-only view.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => a.ReadOnlyView.Slice(0, 2_000_000_000, 1, 2_000_000_000));
    }

    /// <summary>
    /// Copying between overlapping windows must not destroy the source. Column
    /// by column, writing destination column j overwrites source column j+1
    /// before it is read; Span.CopyTo only protects within one column.
    /// </summary>
    [Fact]
    public void CopyToIsSafeBetweenOverlappingWindows()
    {
        var a = Matrix.Zeros<double>(2, 4);

        for (int j = 0; j < 4; j++)
            for (int i = 0; i < 2; i++)
                a[i, j] = j * 10 + i;

        // Shift columns 0..1 one place to the right, into columns 1..2.
        a.Slice(0, 0, 2, 2).CopyTo(a.Slice(0, 1, 2, 2));

        Assert.Equal(0.0, a[0, 1]);
        Assert.Equal(1.0, a[1, 1]);
        Assert.Equal(10.0, a[0, 2]);
        Assert.Equal(11.0, a[1, 2]);

        // Untouched either side.
        Assert.Equal(0.0, a[0, 0]);
        Assert.Equal(30.0, a[0, 3]);
    }

    [Fact]
    public void OverlapDetectionSeparatesSharedFromDisjointWindows()
    {
        var a = Matrix.Zeros<double>(4, 4);
        var b = Matrix.Zeros<double>(4, 4);

        Assert.True(a.Slice(0, 0, 4, 2).Overlaps(a.Slice(0, 1, 4, 2)));
        Assert.False(a.Slice(0, 0, 4, 2).Overlaps(a.Slice(0, 2, 4, 2)));
        Assert.False(a.View.Overlaps(b.View));
    }

    [Fact]
    public void CloneIsIndependentAndPacked()
    {
        var a = new Matrix<double>(3, 2, stride: 7);
        a[0, 0] = 1.0;

        Matrix<double> copy = a.Clone();
        copy[0, 0] = 2.0;

        Assert.Equal(1.0, a[0, 0]);
        Assert.Equal(3, copy.Stride);
    }

    /// <summary>
    /// There is no Dispose to race a view against: storage is a GC-tracked
    /// pinned array, alive while any view of it is reachable. This pins that
    /// the type did not quietly grow one back.
    /// </summary>
    [Fact]
    public void MatrixHasNoDisposeToOutlive() =>
        Assert.False(typeof(IDisposable).IsAssignableFrom(typeof(Matrix<double>)));

    // ---- products ----------------------------------------------------------

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(5, 3, 4)]
    [InlineData(17, 19, 23)]
    [InlineData(64, 64, 64)]
    [InlineData(100, 37, 129)]
    public void MultiplyMatchesTheReferenceImplementation(int m, int n, int k)
    {
        Matrix<double> a = RandomMatrix(m, k, seed: 1);
        Matrix<double> b = RandomMatrix(k, n, seed: 2);

        Matrix<double> product = a.Multiply(b);
        var expected = new Matrix<double>(m, n);

        fixed (double* pa = a.View.Buffer)
        fixed (double* pb = b.View.Buffer)
        fixed (double* pe = expected.View.Buffer)
        {
            Reference.Multiply(m, n, k, 1.0, pa, a.Stride, pb, b.Stride, 0.0, pe, expected.Stride);
        }

        Assert.True(MaxDifference(product, expected) < Tolerance);
    }

    [Fact]
    public void MultiplyRejectsNonConformableShapes()
    {
        Matrix<double> a = RandomMatrix(3, 4, seed: 3);
        Matrix<double> b = RandomMatrix(5, 2, seed: 4);

        Assert.Throws<ArgumentException>(() => a.Multiply(b));
    }

    [Fact]
    public void MultiplyIntoAccumulatesWithBeta()
    {
        Matrix<double> a = Matrix.Identity<double>(4);
        Matrix<double> b = RandomMatrix(4, 4, seed: 5);
        Matrix<double> destination = b.Clone();

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
        Matrix<double> a = RandomDiagonallyDominant(n, seed: 11);
        Matrix<double> expected = RandomMatrix(n, 2, seed: 12);
        Matrix<double> b = a.Multiply(expected);

        Matrix<double> x = a.Solve(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    /// <summary>Factoring must leave the caller's matrix alone.</summary>
    [Fact]
    public void FactorLuDoesNotModifyTheInput()
    {
        Matrix<double> a = RandomDiagonallyDominant(16, seed: 13);
        Matrix<double> before = a.Clone();

        LuDecomposition lu = a.FactorLu();

        Assert.Equal(0.0, MaxDifference(a, before));
    }

    [Fact]
    public void FactorizationCanBeReusedAcrossRightHandSides()
    {
        const int n = 32;

        Matrix<double> a = RandomDiagonallyDominant(n, seed: 14);
        LuDecomposition lu = a.FactorLu();

        for (int trial = 0; trial < 3; trial++)
        {
            Matrix<double> expected = RandomMatrix(n, 1, seed: 100 + trial);
            Matrix<double> b = a.Multiply(expected);
            Matrix<double> x = lu.Solve(b);

            Assert.True(MaxDifference(x, expected) < 1e-9);
        }
    }

    [Fact]
    public void SolveTransposedRecoversAKnownSolution()
    {
        const int n = 24;

        Matrix<double> a = RandomDiagonallyDominant(n, seed: 15);
        LuDecomposition lu = a.FactorLu();

        Matrix<double> expected = RandomMatrix(n, 2, seed: 16);

        // b := A^T * expected, built by multiplying with an explicit transpose.
        Matrix<double> transpose = Transpose(a);
        Matrix<double> b = transpose.Multiply(expected);

        Matrix<double> x = lu.SolveTransposed(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    [Fact]
    public void SolveRejectsANonSquareMatrix()
    {
        Matrix<double> a = RandomMatrix(4, 3, seed: 17);
        Matrix<double> b = RandomMatrix(4, 1, seed: 18);

        Assert.Throws<ArgumentException>(() => a.Solve(b));
    }

    [Fact]
    public void DeterminantMatchesAHandComputation()
    {
        // [[1,2],[3,4]] has determinant -2.
        var a = Matrix.FromRows(new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } });
        LuDecomposition lu = a.FactorLu();

        Assert.Equal(-2.0, lu.Determinant(), 12);
    }

    [Fact]
    public void DeterminantOfIdentityIsOne()
    {
        Matrix<double> a = Matrix.Identity<double>(9);
        LuDecomposition lu = a.FactorLu();

        Assert.Equal(1.0, lu.Determinant(), 12);
    }

    [Fact]
    public void ReciprocalConditionIsOneForTheIdentity()
    {
        Matrix<double> a = Matrix.Identity<double>(32);
        LuDecomposition lu = a.FactorLu();

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
            var a = new Matrix<double>(n, n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    a[i, j] = 1.0 / (i + j + 1);

            LuDecomposition lu = a.FactorLu();
            double rcond = lu.ReciprocalCondition();

            Assert.True(rcond < previous, $"rcond at n={n} was {rcond:E3}, not below {previous:E3}");
            previous = rcond;
        }
    }

    /// <summary>
    /// The zero-copy path: factoring through a workspace overwrites the
    /// operand, the decomposition shares its storage, and it solves exactly as
    /// the copying path does.
    /// </summary>
    [Fact]
    public void WorkspaceFactorsInPlaceAndSharesStorage()
    {
        const int n = 24;

        Matrix<double> a = RandomDiagonallyDominant(n, seed: 31);
        Matrix<double> b = RandomMatrix(n, 2, seed: 32);
        Matrix<double> expected = a.FactorLu().Solve(b);

        Matrix<double> original = a.Clone();
        LuDecomposition lu = Workspace.Shared.FactorLu(a);

        // a now holds the packed factors, not the original.
        Assert.True(MaxDifference(a, original) > 1e-3);
        Assert.Equal(lu.Upper.View[0, 0], a[0, 0]);

        Assert.True(MaxDifference(lu.Solve(b), expected) < Tolerance);
    }

    [Fact]
    public void WorkspaceFactorLuRejectsNull() =>
        Assert.Throws<ArgumentNullException>(() => Workspace.Shared.FactorLu(null!));

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

        Matrix<double> a = RandomDiagonallyDominant(n, seed: 21);
        Matrix<double> b = RandomMatrix(n, 2, seed: 22);

        LuDecomposition lu = a.FactorLu();
        Matrix<double> expected = lu.Solve(b);

        // P*b, then forward substitution through L, then back substitution
        // through U -- each dispatched by its structure type.
        Matrix<double> x = b.Clone();
        fixed (double* px = x.View.Buffer)
        {
            Lu.SwapRows(px, x.Stride, 0, x.Columns, lu.Pivots, 0, n);
        }

        lu.Lower.SolveInPlace(x.View);
        lu.Upper.SolveInPlace(x.View);

        Assert.True(MaxDifference(x, expected) < Tolerance);
    }

    [Fact]
    public void UpperTriangularSolveInvertsMultiplication()
    {
        const int n = 20;

        Matrix<double> u = UpperTriangularMatrix(n, seed: 23);
        Matrix<double> expected = RandomMatrix(n, 3, seed: 24);
        Matrix<double> b = u.Multiply(expected);

        Matrix<double> x = u.As<UpperTriangular>().Solve(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    [Fact]
    public void LowerTriangularSolveInvertsMultiplication()
    {
        const int n = 20;

        Matrix<double> l = LowerTriangularMatrix(n, seed: 25);
        Matrix<double> expected = RandomMatrix(n, 3, seed: 26);
        Matrix<double> b = l.Multiply(expected);

        Matrix<double> x = l.As<LowerTriangular>().Solve(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    [Fact]
    public void TransposedTriangularSolveInvertsMultiplication()
    {
        const int n = 18;

        Matrix<double> u = UpperTriangularMatrix(n, seed: 27);
        Matrix<double> expected = RandomMatrix(n, 2, seed: 28);

        Matrix<double> transpose = Transpose(u);
        Matrix<double> b = transpose.Multiply(expected);

        Matrix<double> x = u.As<UpperTriangular>().SolveTransposed(b);

        Assert.True(MaxDifference(x, expected) < 1e-9);
    }

    /// <summary>
    /// A structure claim describes which triangle is read, not that the rest is
    /// zero. Packed LU storage is the case that forces the distinction.
    /// </summary>
    [Fact]
    public void PackedFactorsDoNotClaimTheirOtherTriangleIsZero()
    {
        Matrix<double> a = RandomDiagonallyDominant(12, seed: 29);
        LuDecomposition lu = a.FactorLu();

        Assert.False(UpperTriangular.UnreferencedPartIsZero(lu.Upper.View));
        Assert.False(UnitLowerTriangular.UnreferencedPartIsZero(lu.Lower.View));

        // ... yet solving through them is still correct, because neither reads
        // the other's triangle. Covered by TriangularFactorsReproduceTheGeneralSolve.
    }

    [Fact]
    public void AsCheckedAcceptsAGenuineTriangleAndRejectsADenseOne()
    {
        Matrix<double> upper = UpperTriangularMatrix(6, seed: 30);
        Matrix<double> dense = RandomMatrix(6, 6, seed: 31);

        StructuredMatrix<double, UpperTriangular> accepted = upper.AsChecked<UpperTriangular>();
        Assert.Equal(6, accepted.Rows);

        Assert.Throws<ArgumentException>(() => dense.AsChecked<UpperTriangular>());
    }

    [Fact]
    public void TriangularSolveRejectsANonSquareOperand()
    {
        Matrix<double> a = RandomMatrix(4, 3, seed: 32);
        Matrix<double> b = RandomMatrix(4, 1, seed: 33);

        Assert.Throws<ArgumentException>(() => a.As<UpperTriangular>().Solve(b));
    }

    // ---- norms -------------------------------------------------------------

    [Fact]
    public void NormsMatchHandComputations()
    {
        var a = Matrix.FromRows(new[,]
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
            Matrix<double> a = RandomMatrix(n, n, seed: n);

            double exact = a.OneNorm();
            double estimate = a.EstimateOneNorm();

            Assert.True(estimate <= exact * (1.0 + 1e-12), $"n={n}: {estimate:E17} > {exact:E17}");
        }
    }

    [Fact]
    public void EstimateOneNormRejectsANonSquareMatrix()
    {
        Matrix<double> a = RandomMatrix(4, 3, seed: 34);

        Assert.Throws<ArgumentException>(() => a.EstimateOneNorm());
    }

    // ---- workspace ---------------------------------------------------------

    [Fact]
    public void AnExplicitWorkspaceAgreesWithTheSharedOne()
    {
        Matrix<double> a = RandomMatrix(48, 48, seed: 41);
        Matrix<double> b = RandomMatrix(48, 48, seed: 42);

        using var workspace = new Workspace(multithreaded: false);

        Matrix<double> viaShared = a.Multiply(b);
        Matrix<double> viaOwn = a.Multiply(b, workspace);

        Assert.True(MaxDifference(viaShared, viaOwn) < Tolerance);
        Assert.False(workspace.IsMultithreaded);
        Assert.NotEmpty(workspace.KernelName);
    }

    [Fact]
    public void DisposedWorkspaceThrows()
    {
        var workspace = new Workspace();
        workspace.Dispose();

        Matrix<double> a = Matrix.Identity<double>(2);
        Matrix<double> b = Matrix.Identity<double>(2);

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
