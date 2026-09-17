namespace Tensile.Tests.Invariants;

/// <summary>
/// I5, the half that hostile-input tests do not cover: when an operation
/// rejects its arguments, it must have written nothing. A destination that is
/// half-overwritten before the shape check fires is a corruption of the
/// caller's data even though no memory was unsafe.
///
/// The technique is a sentinel: fill the destination with a value no
/// computation would produce, provoke the throw, and assert every element is
/// still the sentinel. Green today for the ergonomic layer, which validates
/// before it dispatches; kept so that a refactor cannot move a check to after
/// the first write.
/// </summary>
public class ExceptionSafetyTests
{
    private const double Sentinel = -98765.4321;

    [Fact]
    public void MultiplyIntoWritesNothingWhenShapesDisagree()
    {
        var a = Matrix.Zeros<double>(3, 4);
        var b = Matrix.Zeros<double>(5, 2);          // inner dimensions disagree
        var destination = Matrix.Zeros<double>(3, 2);
        destination.Fill(Sentinel);

        Hostile.MustRejectAsArgument(() => a.MultiplyInto(b, destination.View), "I5", "already green");

        AssertAllSentinel(destination);
    }

    [Fact]
    public void MultiplyIntoWritesNothingWhenDestinationIsWrong()
    {
        var a = Matrix.Zeros<double>(3, 4);
        var b = Matrix.Zeros<double>(4, 2);
        var destination = Matrix.Zeros<double>(3, 3);   // should be 3x2
        destination.Fill(Sentinel);

        Hostile.MustRejectAsArgument(() => a.MultiplyInto(b, destination.View), "I5", "already green");

        AssertAllSentinel(destination);
    }

    [Fact]
    public void CopyToWritesNothingOnShapeMismatch()
    {
        var source = Matrix.Zeros<double>(3, 3);
        var destination = Matrix.Zeros<double>(3, 4);
        destination.Fill(Sentinel);

        Hostile.MustRejectAsArgument(() => source.View.CopyTo(destination.View), "I5", "already green");

        AssertAllSentinel(destination);
    }

    [Fact]
    public void SolveInPlaceWritesNothingWhenRightHandSideHasWrongRows()
    {
        var a = Matrix.Identity<double>(4);
        var rhs = Matrix.Zeros<double>(5, 1);            // should have 4 rows
        rhs.Fill(Sentinel);

        var lu = a.FactorLu();

        Hostile.MustRejectAsArgument(() => lu.SolveInPlace(rhs.View), "I5", "already green");

        AssertAllSentinel(rhs);
    }

    [Fact]
    public void TriangularSolveInPlaceWritesNothingWhenOperandIsNotSquare()
    {
        var t = Matrix.Zeros<double>(4, 3);
        var rhs = Matrix.Zeros<double>(4, 1);
        rhs.Fill(Sentinel);

        Hostile.MustRejectAsArgument(() => t.As<UpperTriangular>().SolveInPlace(rhs.View), "I5", "already green");

        AssertAllSentinel(rhs);
    }

    private static void AssertAllSentinel(Matrix<double> m)
    {
        for (int j = 0; j < m.Columns; j++)
            for (int i = 0; i < m.Rows; i++)
                Assert.True(m[i, j] == Sentinel, $"[I5] element ({i},{j}) was written before the argument check fired.");
    }
}
