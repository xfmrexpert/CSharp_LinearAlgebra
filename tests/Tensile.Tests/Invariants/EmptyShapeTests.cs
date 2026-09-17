using System.Diagnostics;

namespace Tensile.Tests.Invariants;

/// <summary>
/// The fuzzer's first finding. A 0 x n matrix is valid -- its extent is zero
/// -- and n may be anything up to int.MaxValue, so an operation that walks
/// columns walks two billion of them to touch nothing. On the unfixed code
/// <c>Matrix.FromColumnMajor&lt;double&gt;(0, 1_546_977_280, [])</c> took 36
/// seconds. That is resource exhaustion (T4) from a 16-byte input.
///
/// Every column walk now returns first on an empty operand. The bound below
/// is generous -- four orders of magnitude above the fixed cost, four below
/// the defect -- so it fails for the defect on any machine and passes for the
/// fix on the slowest CI runner.
/// </summary>
public class EmptyShapeTests
{
    private const int Huge = int.MaxValue;
    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(2);

    [Fact]
    public void EmptyMatrixWithHugeColumnCountIsCheapEverywhere()
    {
        var wide = new Matrix<double>(0, Huge);        // extent 0
        var tall = new Matrix<double>(Huge, 0);        // extent 0
        var stopwatch = Stopwatch.StartNew();

        _ = Matrix.FromColumnMajor<double>(0, Huge, []);
        _ = Matrix.From(wide.ReadOnlyView);
        _ = wide.Clone();
        _ = wide.ToArray();
        wide.Fill(1.0);
        wide.View.CopyTo(wide.View);
        _ = wide.OneNorm();
        _ = wide.InfinityNorm();
        _ = wide.FrobeniusNorm();
        _ = UpperTriangular.UnreferencedPartIsZero(wide.ReadOnlyView);

        // 0 x k times k x 0: an empty result with a huge inner dimension. (The
        // other order would ask for a Huge x Huge result, which is an invalid
        // shape and rejected as one -- validity first, cost second.)
        _ = wide.Multiply(tall);
        Assert.Throws<ArgumentOutOfRangeException>(() => tall.Multiply(wide));

        // A factorization of nothing, and solves against empty right-hand sides.
        LuDecomposition lu = wide.FactorLu();
        Assert.Equal(0, lu.Pivots.Length);
        _ = Matrix.Identity<double>(4).FactorLu().Solve(new Matrix<double>(4, 0));
        Matrix.Identity<double>(4).As<UpperTriangular>().SolveInPlace(new Matrix<double>(4, 0).View);

        Assert.True(stopwatch.Elapsed < Bound, $"empty-shape operations took {stopwatch.Elapsed}");
    }

    /// <summary>An empty inner dimension is not an empty product: C := beta*C still has to happen.</summary>
    [Fact]
    public void EmptyInnerDimensionStillScalesTheDestination()
    {
        var a = new Matrix<double>(3, 0);
        var b = new Matrix<double>(0, 2);
        var c = new Matrix<double>(3, 2);
        c.Fill(4.0);

        a.MultiplyInto(b, c.View, alpha: 1.0, beta: 0.5);

        Assert.All(c.ToArray(), v => Assert.Equal(2.0, v));
    }
}
