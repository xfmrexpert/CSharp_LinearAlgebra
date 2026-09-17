namespace Tensile.Tests.Invariants;

/// <summary>
/// The secure-by-design specification (docs/security-design.md, section 4) as
/// executable tests.
///
/// These were written BEFORE the refactor they describe and committed red on
/// purpose. Each failing test names the invariant it pins and the migration
/// phase that is expected to turn it green; the phases in section 10 are done
/// when this directory passes. A test here that passes today is a regression
/// guard for a property the current code already has.
///
/// Two rules shaped what is and is not in here:
///
/// - Nothing may depend on the host's memory. A shape whose allocation looks
///   affordable -- tens of gigabytes -- may succeed lazily under Linux
///   overcommit and then be touched, or may fail, depending on the machine.
///   Such cases are excluded; the hostile values below are either invalid
///   outright or so large that every allocator refuses them immediately.
///
/// - Nothing may corrupt memory on today's code. A test that triggers a
///   heap overflow is not red, it is a crashed test host. Where the current
///   defect has both a corrupting and a clean-failing form (see
///   <see cref="HostileDimensionTests.NormEstimateRejectsAProbePanelWhoseSizeOverflows"/>),
///   the clean-failing form is exercised and the corrupting one is documented
///   as closed by the same fix.
/// </summary>
internal static class Hostile
{
    /// <summary>
    /// Single values that a dimension, index or count argument must reject or
    /// handle: the extremes, the negatives, and zero.
    /// </summary>
    public static TheoryData<int> Integers => new() { int.MinValue, -1, 0, 1, int.MaxValue };

    /// <summary>
    /// Pairs whose product overflows int and whose byte count, after the wrap,
    /// is astronomically large -- so every allocator refuses them and the
    /// outcome does not depend on the host. Each is an invalid shape under I2
    /// (required extent does not fit int) and must be rejected as one.
    /// </summary>
    public static TheoryData<int, int> OverflowingPairs => new()
    {
        { int.MaxValue, int.MaxValue },
        { int.MaxValue, 2_000_000_000 },
        { 2_000_000_000, 2_000_000_000 },
    };

    /// <summary>
    /// A bad argument must surface as the documented exception family. Anything
    /// else -- OverflowException, OutOfMemoryException, IndexOutOfRange,
    /// AccessViolation -- means the input got past validation and something
    /// downstream noticed instead, which is exactly the failure this
    /// specification exists to rule out.
    /// </summary>
    /// <param name="action">The call under test.</param>
    /// <param name="invariant">Which section-4 invariant this pins, e.g. "I2".</param>
    /// <param name="closedBy">Which migration phase is expected to make this pass.</param>
    public static void MustRejectAsArgument(Action action, string invariant, string closedBy)
    {
        Exception? error = Record.Exception(action);

        Assert.True(
            error is not null,
            $"[{invariant}] expected an ArgumentException; the call succeeded. Closed by {closedBy}.");

        Assert.True(
            error is ArgumentException,
            $"[{invariant}] expected an ArgumentException; got {error!.GetType().Name}: {error.Message}. "
            + $"Closed by {closedBy}.");
    }

    /// <summary>
    /// The call must either succeed or reject its arguments. It must never
    /// surface an exception from below the validation boundary.
    /// </summary>
    public static void MustSucceedOrRejectAsArgument(Action action, string invariant, string closedBy)
    {
        Exception? error = Record.Exception(action);

        if (error is null) return;

        Assert.True(
            error is ArgumentException,
            $"[{invariant}] expected success or an ArgumentException; got {error.GetType().Name}: {error.Message}. "
            + $"Closed by {closedBy}.");
    }
}
