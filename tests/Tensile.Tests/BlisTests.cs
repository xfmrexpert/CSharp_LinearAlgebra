using Tensile.Interop;

namespace Tensile.Tests;

/// <summary>
/// The optional native binding. Most hosts have no BLIS, so most of this is
/// about failing well: a missing library is a null and a reason, never an
/// exception. When one is present, the view-based product is checked against
/// the managed GEMM, since that is the comparison the binding exists for.
/// </summary>
public class BlisTests
{
    [Fact]
    public void TryLoadEitherLoadsOrExplains()
    {
        using Blis? blis = Blis.TryLoad(out string reason);

        if (blis is null)
        {
            Assert.False(string.IsNullOrWhiteSpace(reason));
            return;
        }

        Assert.Equal("", reason);
        Assert.False(string.IsNullOrWhiteSpace(blis.Version));
        Assert.False(string.IsNullOrWhiteSpace(blis.Architecture));
        Assert.Contains(blis.IntegerBits, new[] { 32, 64 });
    }

    [Fact]
    public void ViewMultiplyAgreesWithManagedGemm()
    {
        using Blis? blis = Blis.TryLoad(out string reason);
        Assert.SkipWhen(blis is null, $"BLIS is not available: {reason}");

        const int m = 37, n = 29, k = 41;

        Matrix<double> a = Random(m, k, seed: 1);
        Matrix<double> b = Random(k, n, seed: 2);
        Matrix<double> expected = a.Multiply(b);
        var c = new Matrix<double>(m, n, stride: m + 3);

        blis!.Multiply(a, b, c.View);

        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                Assert.True(Math.Abs(c[i, j] - expected[i, j]) < 1e-12, $"({i},{j})");
    }

    [Fact]
    public void ViewMultiplyRejectsNonConformableShapes()
    {
        using Blis? blis = Blis.TryLoad(out string reason);
        Assert.SkipWhen(blis is null, $"BLIS is not available: {reason}");

        var a = new Matrix<double>(3, 4);
        var b = new Matrix<double>(5, 2);
        var c = new Matrix<double>(3, 2);

        Assert.Throws<ArgumentException>(() => blis!.Multiply(a, b, c.View));
        Assert.Throws<ArgumentException>(() => blis!.Multiply(a, new Matrix<double>(4, 2), new Matrix<double>(3, 3).View));
    }

    private static Matrix<double> Random(int rows, int columns, int seed)
    {
        var matrix = new Matrix<double>(rows, columns);
        var rng = new Random(seed);

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                matrix[i, j] = rng.NextDouble() - 0.5;

        return matrix;
    }
}
