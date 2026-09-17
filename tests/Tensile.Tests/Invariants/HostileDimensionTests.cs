using Tensile.Primitives;

namespace Tensile.Tests.Invariants;

/// <summary>
/// Every public entry point that takes a dimension, index, stride or count is
/// handed the hostile values in <see cref="Hostile"/> and must either succeed
/// or reject them as arguments. It must never surface an exception from below
/// the validation boundary -- an OverflowException or OutOfMemoryException
/// means the bad input got past validation and something downstream tripped
/// over it, which is the failure mode this specification exists to rule out.
///
/// Pins I1 (bounded access), I2 (shape validity), I5 (fail closed) and I7 (no
/// silent wrap). The Slice and CopyTo cases are green today: they are the two
/// review findings that motivated this work, kept as regressions. The
/// allocation-extent cases are red today and are closed by Phase 2, when
/// MatrixShape validates the extent before anything is allocated.
/// </summary>
public class HostileDimensionTests
{
    // ---- I2: shape validity ----------------------------------------------

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void MatrixConstructorHandlesAHostileRowCount(int rows) =>
        Hostile.MustSucceedOrRejectAsArgument(() => new Matrix<double>(rows, 2), "I2", "Phase 2");

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void MatrixConstructorHandlesAHostileColumnCount(int columns) =>
        Hostile.MustSucceedOrRejectAsArgument(() => new Matrix<double>(2, columns), "I2", "Phase 2");

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void MatrixConstructorHandlesAHostileStride(int stride) =>
        Hostile.MustSucceedOrRejectAsArgument(() => new Matrix<double>(2, 2, stride), "I2", "Phase 2");

    /// <summary>
    /// A shape whose required extent does not fit int is invalid, not merely
    /// unaffordable, and must be rejected as an argument before any allocation
    /// is attempted. Today the product wraps nuint on the way to the allocator
    /// and the failure is reported by whichever layer notices -- an
    /// OutOfMemoryException from a 14-exabyte request, or an
    /// OverflowException from the cast that follows it -- rather than by
    /// validation.
    /// </summary>
    [Theory]
    [MemberData(nameof(Hostile.OverflowingPairs), MemberType = typeof(Hostile))]
    public void MatrixConstructorRejectsAShapeWhoseExtentOverflows(int rows, int columns) =>
        Hostile.MustRejectAsArgument(() => new Matrix<double>(rows, columns), "I2/I7", "Phase 2");

    [Fact]
    public void MatrixConstructorRejectsAStrideThatOverflowsTheExtent() =>
        Hostile.MustRejectAsArgument(
            () => new Matrix<double>(1, int.MaxValue, stride: int.MaxValue), "I2/I7", "Phase 2");

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void IdentityHandlesAHostileOrder(int order) =>
        Hostile.MustSucceedOrRejectAsArgument(() => Matrix.Identity<double>(order), "I2", "Phase 2");

    [Theory]
    [MemberData(nameof(Hostile.OverflowingPairs), MemberType = typeof(Hostile))]
    public void ZerosRejectsAShapeWhoseExtentOverflows(int rows, int columns) =>
        Hostile.MustRejectAsArgument(() => Matrix.Zeros<double>(rows, columns), "I2/I7", "Phase 2");

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void FromColumnMajorHandlesAHostileShape(int dimension) =>
        Hostile.MustSucceedOrRejectAsArgument(
            () => Matrix.FromColumnMajor<double>(dimension, dimension, new double[4]), "I2", "Phase 2");

    // ---- I1: bounded access ------------------------------------------------

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void SliceHandlesAHostileRowOrigin(int row)
    {
        var a = Matrix.Zeros<double>(4, 4);
        Hostile.MustSucceedOrRejectAsArgument(() => a.Slice(row, 0, 1, 1), "I1", "Phase 2");
    }

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void SliceHandlesAHostileColumnOrigin(int column)
    {
        var a = Matrix.Zeros<double>(4, 4);
        Hostile.MustSucceedOrRejectAsArgument(() => a.Slice(0, column, 1, 1), "I1", "Phase 2");
    }

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void SliceHandlesAHostileExtent(int extent)
    {
        var a = Matrix.Zeros<double>(4, 4);
        Hostile.MustSucceedOrRejectAsArgument(() => a.Slice(0, 0, extent, extent), "I1", "Phase 2");
    }

    /// <summary>
    /// The pair that produced a 1x2000000000 view 16 GB past a 1x4 matrix.
    /// Fixed in review; kept because it is the canonical instance of the class.
    /// </summary>
    [Fact]
    public void SliceRejectsOriginPlusExtentThatWraps()
    {
        var a = Matrix.Zeros<double>(1, 4);

        Hostile.MustRejectAsArgument(() => a.Slice(0, 2_000_000_000, 1, 2_000_000_000), "I1", "already green");
        Hostile.MustRejectAsArgument(() => a.Slice(2_000_000_000, 0, 2_000_000_000, 1), "I1", "already green");
        Hostile.MustRejectAsArgument(() => a.ReadOnlyView.Slice(0, 2_000_000_000, 1, 2_000_000_000), "I1", "already green");
    }

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void ColumnHandlesAHostileIndex(int index)
    {
        var a = Matrix.Zeros<double>(4, 4);
        Hostile.MustSucceedOrRejectAsArgument(() => a.Column(index), "I1", "Phase 2");
    }

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void IndexerHandlesHostileIndices(int index)
    {
        var a = Matrix.Zeros<double>(4, 4);
        Hostile.MustSucceedOrRejectAsArgument(() => _ = a[index, 0], "I1", "Phase 2");
        Hostile.MustSucceedOrRejectAsArgument(() => _ = a[0, index], "I1", "Phase 2");
    }

    // ---- I5 / I7 through the operations -----------------------------------

    [Fact]
    public void MultiplyRejectsNonConformableShapesAsArguments()
    {
        var a = Matrix.Zeros<double>(3, 4);
        var b = Matrix.Zeros<double>(5, 2);

        Hostile.MustRejectAsArgument(() => a.Multiply(b), "I5", "already green");
    }

    /// <summary>
    /// The audit finding. The estimator computes its probe panel as
    /// <c>n * t</c> in unchecked int. With <c>n = t = 50000</c> that wraps
    /// negative, sign-extends to a 14-exabyte allocation, and surfaces as
    /// OutOfMemoryException -- the wrong diagnosis for what is an invalid
    /// request (a 2.5e9-element panel does not fit int) and must be rejected
    /// as an argument.
    ///
    /// This is the clean-failing form of the defect, chosen so the test is
    /// runnable. The corrupting form is <c>n = t = 65536</c>, where the
    /// product wraps to exactly zero, a zero-length buffer is handed back, and
    /// the estimator writes 2^32 doubles into it. That case is NOT exercised
    /// here because it would take the test host down rather than fail; the
    /// same validation closes both.
    ///
    /// Reachable through the public extension point: an ILinearOperator is
    /// free to report any Order, and a matrix-free operator has no storage
    /// whose size would constrain it.
    /// </summary>
    [Fact]
    public unsafe void NormEstimateRejectsAProbePanelWhoseSizeOverflows()
    {
        var op = new HostileOrderOperator(order: 50_000);

        Hostile.MustRejectAsArgument(
            () => NormEstimate.Of(op, columns: 50_000), "I2/I7", "Phase 2 (checked extent) and Phase 3 (operator over views)");
    }

    /// <summary>Documents that columns is clamped, not rejected, at either extreme.</summary>
    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void EstimateOneNormHandlesAHostileColumnCount(int columns)
    {
        var a = Matrix.Identity<double>(8);
        Hostile.MustSucceedOrRejectAsArgument(() => a.EstimateOneNorm(columns: columns), "I5", "already green");
    }

    [Theory]
    [MemberData(nameof(Hostile.Integers), MemberType = typeof(Hostile))]
    public void FactorLuHandlesAHostileBlockSize(int blockSize)
    {
        var a = Matrix.Identity<double>(8);
        Hostile.MustSucceedOrRejectAsArgument(() => a.FactorLu(blockSize), "I5", "already green");
    }

    /// <summary>
    /// An operator that reports an order it has no storage for. This is not a
    /// contrived attacker: it is the shape of every matrix-free operator, and
    /// the reason the estimator cannot trust Order alone.
    /// </summary>
    private sealed unsafe class HostileOrderOperator(int order) : ILinearOperator
    {
        public int Order => order;

        public void Apply(int t, double* x, int ldx, double* y, int ldy) =>
            throw new InvalidOperationException("Apply must not be reached: validation should have rejected the panel size.");

        public void ApplyTranspose(int t, double* x, int ldx, double* y, int ldy) =>
            throw new InvalidOperationException("ApplyTranspose must not be reached.");
    }
}
