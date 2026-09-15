using System.Runtime.InteropServices;

namespace GemmLab.Tests;

/// <summary>
/// An aligned column-major test matrix.
///
/// The library works in raw pointers, so the tests do too; this wrapper exists
/// only to make allocation, indexing and disposal readable, and to make it easy
/// to give a matrix a column stride larger than its row count, which is where
/// stride bugs surface.
/// </summary>
internal sealed unsafe class Matrix : IDisposable
{
    public int Rows { get; }
    public int Columns { get; }
    public int Stride { get; }

    public double* Data { get; private set; }

    public Matrix(int rows, int columns, int stride = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);

        Rows = rows;
        Columns = columns;
        Stride = stride == 0 ? rows : stride;

        if (Stride < rows) throw new ArgumentOutOfRangeException(nameof(stride));

        int count = Math.Max(1, Stride * columns);
        Data = (double*)NativeMemory.AlignedAlloc((nuint)count * sizeof(double), 64);
        new Span<double>(Data, count).Clear();
    }

    public int Count => Stride * Columns;

    public double this[int row, int column]
    {
        get => Data[(nint)column * Stride + row];
        set => Data[(nint)column * Stride + row] = value;
    }

    /// <summary>Uniform random entries in [-0.5, 0.5], padding left at zero.</summary>
    public static Matrix Random(int rows, int columns, int seed, int stride = 0)
    {
        var matrix = new Matrix(rows, columns, stride);
        var rng = new Random(seed);

        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                matrix[i, j] = rng.NextDouble() - 0.5;

        return matrix;
    }

    /// <summary>
    /// Random entries with a diagonal large enough that the matrix is well
    /// conditioned and needs no pivoting to stay accurate.
    /// </summary>
    public static Matrix RandomDiagonallyDominant(int n, int seed, int stride = 0)
    {
        var matrix = Random(n, n, seed, stride);
        for (int i = 0; i < n; i++) matrix[i, i] += n;
        return matrix;
    }

    /// <summary>Fill every slot including padding, so overwrites of padding are visible.</summary>
    public void FillAll(double value) => new Span<double>(Data, Count).Fill(value);

    public Matrix Clone()
    {
        var copy = new Matrix(Rows, Columns, Stride);
        new Span<double>(Data, Count).CopyTo(new Span<double>(copy.Data, Count));
        return copy;
    }

    /// <summary>Largest absolute difference over the logical (non-padding) entries.</summary>
    public double MaxDifference(Matrix other)
    {
        double worst = 0.0;

        for (int j = 0; j < Columns; j++)
            for (int i = 0; i < Rows; i++)
                worst = Math.Max(worst, Math.Abs(this[i, j] - other[i, j]));

        return worst;
    }

    public void Dispose()
    {
        if (Data is not null) { NativeMemory.AlignedFree(Data); Data = null; }
        GC.SuppressFinalize(this);
    }

    ~Matrix() => Dispose();
}
