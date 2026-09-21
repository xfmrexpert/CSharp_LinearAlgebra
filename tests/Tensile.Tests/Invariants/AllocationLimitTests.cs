namespace Tensile.Tests.Invariants;

/// <summary>
/// The limit is a process-wide static, so these tests cannot share a process
/// with anything allocating concurrently. The collection is serialised against
/// the rest of the suite, and every test restores the default on exit.
/// </summary>
[CollectionDefinition(nameof(TensileLimitsCollection), DisableParallelization = true)]
public sealed class TensileLimitsCollection;

/// <summary>
/// I9: allocation goes through one path with a configurable ceiling. A request
/// over the ceiling is refused before any memory is asked for, with an
/// exception that names both the request and the limit, and it is a different
/// exception from the one the runtime throws when it tries and fails.
///
/// Order matters and is pinned: shape validity (I2) is checked before policy,
/// so an invalid shape is still an argument error under any limit. The limit
/// only ever sees shapes that could exist.
/// </summary>
[Collection(nameof(TensileLimitsCollection))]
public sealed class AllocationLimitTests : IDisposable
{
    public void Dispose() => TensileLimits.Reset();

    [Fact]
    public void DefaultIsTheRuntimeCeiling()
    {
        TensileLimits.Reset();
        Assert.Equal(Array.MaxLength, TensileLimits.MaxElements);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(int.MinValue)]
    [InlineData(int.MaxValue)]                // above Array.MaxLength
    public void SetterRejectsAnUnusableLimit(int value)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => TensileLimits.MaxElements = value);
        Assert.Equal(Array.MaxLength, TensileLimits.MaxElements);
    }

    [Fact]
    public void MatrixOverTheLimitIsRefusedWithBothNumbers()
    {
        TensileLimits.MaxElements = 1000;

        var error = Assert.Throws<AllocationLimitException>(() => new Matrix<double>(40, 40));

        Assert.Equal(1600, error.Requested);
        Assert.Equal(1000, error.Limit);
        Assert.Contains("1600", error.Message);
        Assert.Contains("1000", error.Message);
    }

    /// <summary>The limit is on the request, not on what is allocated: alignment padding is the library's cost.</summary>
    [Fact]
    public void MatrixExactlyAtTheLimitIsAllowed()
    {
        TensileLimits.MaxElements = 1600;

        var a = new Matrix<double>(40, 40);

        Assert.Equal(1600, a.Shape.RequiredExtent);
    }

    /// <summary>
    /// It is the required extent that is measured, padding included: a strided
    /// matrix asks for more than rows * columns, and that is what it gets.
    /// </summary>
    [Fact]
    public void StridePaddingCountsTowardTheLimit()
    {
        TensileLimits.MaxElements = 1600;

        Assert.Throws<AllocationLimitException>(() => new Matrix<double>(40, 40, stride: 41));
    }

    [Fact]
    public void LimitIsElementsNotBytes()
    {
        TensileLimits.MaxElements = 1000;

        Assert.Throws<AllocationLimitException>(() => new Matrix<Half>(40, 40));
        Assert.Throws<AllocationLimitException>(() => new Matrix<double>(40, 40));
        _ = new Matrix<double>(31, 31);
    }

    [Fact]
    public void InvalidShapeIsStillAnArgumentErrorUnderAnyLimit()
    {
        TensileLimits.MaxElements = 1000;

        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix<double>(int.MaxValue, int.MaxValue));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Matrix<double>(-1, 4));
    }

    [Fact]
    public void ResultAllocationsAreCoveredToo()
    {
        var a = new Matrix<double>(40, 25);
        var b = new Matrix<double>(25, 40);

        TensileLimits.MaxElements = 999;

        // The operands already exist and stay usable; every copy or result the
        // library would now allocate from them is over the line.
        Assert.Throws<AllocationLimitException>(() => a.Multiply(b));
        Assert.Throws<AllocationLimitException>(() => a.Clone());
        Assert.Throws<AllocationLimitException>(() => a.View.ToArray());
        Assert.Throws<AllocationLimitException>(() => Matrix.From(a.ReadOnlyView));
    }

    /// <summary>
    /// A view bound over a caller's own array was never allocated by the
    /// library, but anything the library allocates from it is still policed.
    /// </summary>
    [Fact]
    public void CopiesOfACallerBoundViewAreCovered()
    {
        double[] mine = new double[2000];
        TensileLimits.MaxElements = 1000;

        // A ref struct cannot be captured by a lambda, so the view is rebound
        // inside the call under test.
        Assert.Throws<AllocationLimitException>(
            () => MatrixView<double>.Bind(mine, new MatrixShape(50, 40)).ToArray());

        // Reading and writing through the view itself needs no allocation.
        MatrixView<double> view = MatrixView<double>.Bind(mine, new MatrixShape(50, 40));
        view[49, 39] = 1.0;
        Assert.Equal(1.0, mine[1999]);
    }

    /// <summary>
    /// The estimator's probe panel is sized by an operator's self-reported
    /// order, which a matrix-free operator can set to anything. The panel is
    /// what the limit measures.
    /// </summary>
    [Fact]
    public void EstimatorProbePanelIsCovered()
    {
        TensileLimits.MaxElements = 500;

        var op = new UntouchedOperator(order: 1000);

        var error = Assert.Throws<AllocationLimitException>(() => NormEstimate.Of(op, columns: 1));

        Assert.Equal(1000, error.Requested);
        Assert.False(op.Touched, "the operator must not be applied when its panel was refused");
    }

    /// <summary>Refusal leaves nothing behind: the next request under the limit succeeds normally.</summary>
    [Fact]
    public void RefusalHasNoLastingEffect()
    {
        TensileLimits.MaxElements = 1000;

        Assert.Throws<AllocationLimitException>(() => new Matrix<double>(40, 40));

        var a = Matrix.Identity<double>(30);
        Assert.Equal(30.0, a.OneNorm() * 30);
        Assert.Equal(1000, TensileLimits.MaxElements);
    }

    [Fact]
    public void ResetRestoresTheDefault()
    {
        TensileLimits.MaxElements = 1;
        TensileLimits.Reset();

        Assert.Equal(Array.MaxLength, TensileLimits.MaxElements);
        _ = new Matrix<double>(100, 100);
    }

    private sealed class UntouchedOperator(int order) : ILinearOperator
    {
        public bool Touched { get; private set; }

        public int Order => order;

        public void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y) => Touched = true;

        public void ApplyTranspose(ReadOnlyMatrixView<double> x, MatrixView<double> y) => Touched = true;
    }
}
