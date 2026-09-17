using Tensile.Primitives;

namespace Tensile.Tests;

/// <summary>
/// Triangular solves, checked by substituting the solution back.
///
/// The transposed solves are the ones that most need this. They read the same
/// packed LU storage as the untransposed pair but walk it in the other
/// direction, and the 1-norm condition estimator is their only caller -- an
/// estimator which, being a heuristic, would happily return a plausible number
/// from a wrong solve.
/// </summary>
public unsafe class TriangularTests
{
    private const double Tolerance = 1e-11;

    public static TheoryData<int, int> Shapes => new()
    {
        { 1, 1 }, { 2, 1 }, { 5, 3 }, { 8, 4 }, { 16, 1 }, { 17, 6 }, { 33, 4 }, { 64, 7 }, { 65, 9 },
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SolveLowerUnitInvertsMultiplication(int m, int nrhs)
    {
        using var l = UnitLower(m, seed: 1);
        using var x = TestMatrix.Random(m, nrhs, seed: 2);
        using var b = Multiply(l, x, unitLower: true, transposed: false);

        Triangular.SolveLowerUnit(m, nrhs, l.Data, l.Stride, b.Data, b.Stride);

        Assert.True(b.MaxDifference(x) < Tolerance, $"worst {b.MaxDifference(x):E3}");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SolveUpperInvertsMultiplication(int m, int nrhs)
    {
        using var u = Upper(m, seed: 3);
        using var x = TestMatrix.Random(m, nrhs, seed: 4);
        using var b = Multiply(u, x, unitLower: false, transposed: false);

        Triangular.SolveUpper(m, nrhs, u.Data, u.Stride, b.Data, b.Stride);

        Assert.True(b.MaxDifference(x) < Tolerance, $"worst {b.MaxDifference(x):E3}");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SolveLowerUnitTransposedInvertsMultiplication(int m, int nrhs)
    {
        using var l = UnitLower(m, seed: 5);
        using var x = TestMatrix.Random(m, nrhs, seed: 6);
        using var b = Multiply(l, x, unitLower: true, transposed: true);

        Triangular.SolveLowerUnitTransposed(m, nrhs, l.Data, l.Stride, b.Data, b.Stride);

        Assert.True(b.MaxDifference(x) < Tolerance, $"worst {b.MaxDifference(x):E3}");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void SolveUpperTransposedInvertsMultiplication(int m, int nrhs)
    {
        using var u = Upper(m, seed: 7);
        using var x = TestMatrix.Random(m, nrhs, seed: 8);
        using var b = Multiply(u, x, unitLower: false, transposed: true);

        Triangular.SolveUpperTransposed(m, nrhs, u.Data, u.Stride, b.Data, b.Stride);

        Assert.True(b.MaxDifference(x) < Tolerance, $"worst {b.MaxDifference(x):E3}");
    }

    /// <summary>
    /// The solves must ignore whatever sits in the unreferenced triangle, since
    /// LU packs L and U into one array and each solve sees the other's data
    /// there.
    /// </summary>
    [Fact]
    public void UnreferencedTriangleIsIgnored()
    {
        const int m = 24, nrhs = 3;

        using var l = UnitLower(m, seed: 9);
        using var x = TestMatrix.Random(m, nrhs, seed: 10);
        using var b = Multiply(l, x, unitLower: true, transposed: false);

        // Poison the strict upper triangle and the implicit unit diagonal.
        for (int j = 0; j < m; j++)
            for (int i = 0; i <= j; i++)
                l[i, j] = double.NaN;

        Triangular.SolveLowerUnit(m, nrhs, l.Data, l.Stride, b.Data, b.Stride);

        Assert.True(b.MaxDifference(x) < Tolerance);
    }

    /// <summary>Unit lower triangular with a bounded strict lower part.</summary>
    private static TestMatrix UnitLower(int m, int seed)
    {
        var matrix = new TestMatrix(m, m, stride: m + 2);
        var rng = new Random(seed);

        for (int j = 0; j < m; j++)
            for (int i = j + 1; i < m; i++)
                matrix[i, j] = rng.NextDouble() - 0.5;

        return matrix;
    }

    /// <summary>Upper triangular with a diagonal safely away from zero.</summary>
    private static TestMatrix Upper(int m, int seed)
    {
        var matrix = new TestMatrix(m, m, stride: m + 2);
        var rng = new Random(seed);

        for (int j = 0; j < m; j++)
        {
            for (int i = 0; i < j; i++) matrix[i, j] = rng.NextDouble() - 0.5;
            matrix[j, j] = 1.0 + rng.NextDouble();
        }

        return matrix;
    }

    /// <summary>
    /// B := T*X or T^T*X, computed the naive way so it is independent of
    /// anything under test.
    /// </summary>
    private static TestMatrix Multiply(TestMatrix t, TestMatrix x, bool unitLower, bool transposed)
    {
        int m = x.Rows;
        var b = new TestMatrix(m, x.Columns, stride: m + 3);

        for (int col = 0; col < x.Columns; col++)
        {
            for (int i = 0; i < m; i++)
            {
                double sum = 0.0;

                for (int j = 0; j < m; j++)
                {
                    // Entry (i,j) of the operator being applied.
                    double entry = transposed ? Entry(t, j, i, unitLower) : Entry(t, i, j, unitLower);
                    sum += entry * x[j, col];
                }

                b[i, col] = sum;
            }
        }

        return b;
    }

    private static double Entry(TestMatrix t, int row, int column, bool unitLower)
    {
        if (unitLower)
        {
            if (row == column) return 1.0;
            return row > column ? t[row, column] : 0.0;
        }

        return row <= column ? t[row, column] : 0.0;
    }
}
