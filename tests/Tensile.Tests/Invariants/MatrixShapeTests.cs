namespace Tensile.Tests.Invariants;

/// <summary>
/// I2 at its source: a <see cref="MatrixShape"/> cannot exist unless valid, and
/// every derived quantity -- extent, offset, sub-shape -- is checked. Plus the
/// binding step that turns a shape and a span into a view (I1), which is the
/// only public route to a view and therefore the place a too-short buffer must
/// be caught.
///
/// These are the Phase 2 behavioural tests the design document said could not
/// be committed in Phase 1 because their types did not compile yet.
/// </summary>
public class MatrixShapeTests
{
    // ---- construction ------------------------------------------------------

    [Theory]
    [InlineData(-1, 2, 0)]
    [InlineData(2, -1, 0)]
    [InlineData(int.MinValue, 2, 0)]
    [InlineData(2, 2, 1)]                  // stride below rows
    [InlineData(2, 2, -5)]
    public void ConstructorRejectsInvalidDimensions(int rows, int columns, int stride) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new MatrixShape(rows, columns, stride));

    [Theory]
    [MemberData(nameof(Hostile.OverflowingPairs), MemberType = typeof(Hostile))]
    public void ConstructorRejectsAnExtentThatDoesNotFitInt(int rows, int columns) =>
        Hostile.MustRejectAsArgument(() => new MatrixShape(rows, columns), "I2", "Phase 2");

    /// <summary>The stride path to the same overflow: few columns, enormous stride.</summary>
    [Fact]
    public void ConstructorRejectsAStrideThatOverflowsTheExtent() =>
        Hostile.MustRejectAsArgument(() => new MatrixShape(1, int.MaxValue, stride: int.MaxValue), "I2", "Phase 2");

    [Fact]
    public void ZeroStrideMeansPacked()
    {
        var shape = new MatrixShape(3, 4);

        Assert.Equal(3, shape.Stride);
        Assert.True(shape.IsContiguous);
    }

    [Fact]
    public void EmptyShapesAreValid()
    {
        Assert.Equal(0, new MatrixShape(0, 5).RequiredExtent);
        Assert.Equal(0, new MatrixShape(5, 0).RequiredExtent);
        Assert.True(new MatrixShape(0, 0).IsEmpty);
    }

    // ---- extent ------------------------------------------------------------

    /// <summary>
    /// The tight bound: the last column needs no trailing padding, so a
    /// caller's exactly-sized buffer binds.
    /// </summary>
    [Theory]
    [InlineData(3, 4, 3, 12)]      // packed: 3*4
    [InlineData(3, 4, 5, 18)]      // (4-1)*5 + 3, not 5*4 = 20
    [InlineData(1, 1, 1, 1)]
    [InlineData(7, 1, 100, 7)]     // one column: stride irrelevant
    public void RequiredExtentIsTheTightBound(int rows, int columns, int stride, int expected) =>
        Assert.Equal(expected, new MatrixShape(rows, columns, stride).RequiredExtent);

    /// <summary>
    /// The largest shape that is still valid: extent exactly int.MaxValue.
    /// One past it must fail. This pins the boundary rather than a point well
    /// inside it.
    /// </summary>
    [Fact]
    public void ExtentBoundaryIsExact()
    {
        // 1 row, int.MaxValue columns, stride 1 -> extent int.MaxValue. Valid.
        var largest = new MatrixShape(1, int.MaxValue, stride: 1);
        Assert.Equal(int.MaxValue, largest.RequiredExtent);

        // 2 rows, int.MaxValue columns, stride 2 -> extent 2^32 - 2. Invalid.
        Assert.Throws<ArgumentOutOfRangeException>(() => new MatrixShape(2, int.MaxValue, stride: 2));
    }

    // ---- offsets -----------------------------------------------------------

    [Fact]
    public void OffsetOfFollowsColumnMajorLayout()
    {
        var shape = new MatrixShape(3, 4, stride: 5);

        Assert.Equal(0, shape.OffsetOf(0, 0));
        Assert.Equal(2, shape.OffsetOf(2, 0));
        Assert.Equal(5, shape.OffsetOf(0, 1));
        Assert.Equal(17, shape.OffsetOf(2, 3));           // 3*5 + 2 == RequiredExtent - 1
        Assert.Equal(shape.RequiredExtent - 1, shape.OffsetOf(2, 3));
    }

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void OffsetOfRejectsHostileIndices(int index)
    {
        var shape = new MatrixShape(3, 4);

        if (index is >= 0 and < 3) _ = shape.OffsetOf(index, 0);
        else Assert.Throws<ArgumentOutOfRangeException>(() => shape.OffsetOf(index, 0));

        if (index is >= 0 and < 4) _ = shape.OffsetOf(0, index);
        else Assert.Throws<ArgumentOutOfRangeException>(() => shape.OffsetOf(0, index));
    }

    /// <summary>A shape with no rows still has addressable column starts.</summary>
    [Fact]
    public void ColumnOffsetWorksWithZeroRows()
    {
        var shape = new MatrixShape(0, 3, stride: 0);

        Assert.Equal(0, shape.ColumnOffset(2));
    }

    // ---- sub-shapes --------------------------------------------------------

    [Fact]
    public void SubKeepsTheParentStride()
    {
        var parent = new MatrixShape(10, 10, stride: 12);
        MatrixShape sub = parent.Sub(2, 3, 4, 5);

        Assert.Equal(4, sub.Rows);
        Assert.Equal(5, sub.Columns);
        Assert.Equal(12, sub.Stride);
    }

    /// <summary>
    /// The review finding, at the shape level: origin + extent that wraps int
    /// must be rejected, and checked by subtraction so it cannot itself wrap.
    /// </summary>
    [Fact]
    public void SubRejectsOriginPlusExtentThatWraps()
    {
        var parent = new MatrixShape(1, 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => parent.Sub(0, 2_000_000_000, 1, 2_000_000_000));
        Assert.Throws<ArgumentOutOfRangeException>(() => parent.Sub(2_000_000_000, 0, 2_000_000_000, 1));
    }

    [Fact]
    public void SubAllowsAnEmptyBlockAtTheFarEdge()
    {
        var parent = new MatrixShape(4, 4);

        MatrixShape atBottom = parent.Sub(4, 0, 0, 2);
        MatrixShape atRight = parent.Sub(0, 4, 2, 0);

        Assert.True(atBottom.IsEmpty);
        Assert.True(atRight.IsEmpty);
    }

    [Fact]
    public void SubRejectsABlockThatLeavesTheParent()
    {
        var parent = new MatrixShape(4, 4);

        Assert.Throws<ArgumentOutOfRangeException>(() => parent.Sub(2, 2, 3, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => parent.Sub(0, 0, 5, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => parent.Sub(5, 0, 0, 0));
    }

    // ---- binding (I1) ------------------------------------------------------

    [Fact]
    public void BindRejectsABufferShorterThanTheExtent()
    {
        var shape = new MatrixShape(3, 4, stride: 5);          // needs 18
        double[] buffer = new double[17];

        Hostile.MustRejectAsArgument(() => MatrixView<double>.Bind(buffer, shape), "I1", "Phase 2");
        Hostile.MustRejectAsArgument(() => ReadOnlyMatrixView<double>.Bind(buffer, shape), "I1", "Phase 2");
    }

    [Fact]
    public void BindAcceptsAnExactlySizedBuffer()
    {
        var shape = new MatrixShape(3, 4, stride: 5);
        double[] buffer = new double[18];

        MatrixView<double> view = MatrixView<double>.Bind(buffer, shape);

        Assert.Equal(shape, view.Shape);
        Assert.Equal(18, view.Column(3).Length + shape.OffsetOf(0, 3));   // last column ends at the extent
    }

    /// <summary>
    /// Binding over an ordinary managed array -- the zero-copy case the
    /// pointer constructor used to serve -- and writes go where the shape says.
    /// </summary>
    [Fact]
    public void BindOverACallerArrayWritesThrough()
    {
        double[] buffer = new double[6];
        MatrixView<double> view = MatrixView<double>.Bind(buffer, new MatrixShape(2, 3));

        view[1, 2] = 7.0;

        Assert.Equal(7.0, buffer[5]);        // column-major: 2*2 + 1
    }

    /// <summary>
    /// The runtime as last line of defence: a slice of a bound view is a
    /// Span.Slice, so a wrong offset throws rather than reading past the array.
    /// Exercised through the public path since the arithmetic is private.
    /// </summary>
    [Fact]
    public void SlicesOfABoundViewStayWithinTheBuffer()
    {
        double[] buffer = new double[18];
        MatrixView<double> view = MatrixView<double>.Bind(buffer, new MatrixShape(3, 4, stride: 5));

        MatrixView<double> corner = view.Slice(1, 2, 2, 2);

        corner.Fill(1.0);

        // Only the four addressed elements changed, and nothing in the padding.
        Assert.Equal(4, buffer.Count(v => v == 1.0));
        Assert.Equal(1.0, buffer[new MatrixShape(3, 4, stride: 5).OffsetOf(2, 3)]);
    }
}
