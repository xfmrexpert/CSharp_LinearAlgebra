using System.Numerics;
using System.Runtime.InteropServices;

namespace Tensile;

/// <summary>
/// A dense column-major matrix that owns its storage.
///
/// Storage is native and 64-byte aligned rather than a <c>T[]</c>, for two
/// reasons: the micro-kernels issue aligned vector loads against it, and native
/// memory does not move, so a <see cref="MatrixView{T}"/> handed to a kernel
/// stays valid without pinning. The cost is that instances are disposable and a
/// dropped reference leaks until finalization.
///
/// The type is generic so that storage, views, slicing and the structure
/// vocabulary are written once. Arithmetic is currently supplied only for
/// <see cref="double"/>, through extension methods on the closed type, which is
/// why <c>Matrix&lt;float&gt;</c> will compile and hold data but has nothing to
/// multiply it with yet. Adding a numeric type is then additive: the signatures
/// here do not change.
/// </summary>
/// <typeparam name="T">Element type.</typeparam>
public sealed unsafe class Matrix<T> : IDisposable where T : unmanaged, INumberBase<T>
{
    private T* _data;

    /// <summary>Allocate a zeroed matrix.</summary>
    /// <param name="rows">Row count.</param>
    /// <param name="columns">Column count.</param>
    /// <param name="stride">
    /// Column stride. Zero means "packed", that is equal to
    /// <paramref name="rows"/>. A larger value leaves padding between columns,
    /// which is mainly useful for reproducing a caller's layout or for testing
    /// that operations respect stride.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">A dimension is negative, or the stride is below the row count.</exception>
    public Matrix(int rows, int columns, int stride = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);

        if (stride == 0) stride = rows;
        ArgumentOutOfRangeException.ThrowIfLessThan(stride, rows);

        Rows = rows;
        Columns = columns;
        Stride = stride;

        nuint count = Math.Max(1, (nuint)stride * (nuint)columns);
        _data = (T*)NativeMemory.AlignedAlloc(count * (nuint)sizeof(T), 64);
        new Span<T>(_data, checked((int)count)).Clear();
    }

    /// <summary>Row count.</summary>
    public int Rows { get; }

    /// <summary>Column count.</summary>
    public int Columns { get; }

    /// <summary>Distance in elements between the starts of consecutive columns.</summary>
    public int Stride { get; }

    /// <summary>Whether the matrix has no elements.</summary>
    public bool IsEmpty => Rows == 0 || Columns == 0;

    /// <summary>Whether the matrix is square, and so a candidate for solving and factorization.</summary>
    public bool IsSquare => Rows == Columns;

    /// <summary>Element (<paramref name="row"/>, <paramref name="column"/>), by reference.</summary>
    /// <exception cref="ObjectDisposedException">The matrix has been disposed.</exception>
    /// <exception cref="ArgumentOutOfRangeException">Either index is out of range.</exception>
    public ref T this[int row, int column] => ref View[row, column];

    /// <summary>A writable view over the whole matrix.</summary>
    /// <exception cref="ObjectDisposedException">The matrix has been disposed.</exception>
    public MatrixView<T> View
    {
        get
        {
            ObjectDisposedException.ThrowIf(_data is null, this);
            return new MatrixView<T>(_data, Rows, Columns, Stride);
        }
    }

    /// <summary>A read-only view over the whole matrix.</summary>
    /// <exception cref="ObjectDisposedException">The matrix has been disposed.</exception>
    public ReadOnlyMatrixView<T> ReadOnlyView => View;

    /// <summary>One column as a span, which is contiguous and therefore free.</summary>
    /// <param name="index">Column index.</param>
    /// <exception cref="ArgumentOutOfRangeException">The index is out of range.</exception>
    public Span<T> Column(int index) => View.Column(index);

    /// <summary>A view onto a block of this matrix. Shares storage; copies nothing.</summary>
    /// <param name="row">First row of the block.</param>
    /// <param name="column">First column of the block.</param>
    /// <param name="rows">Rows in the block.</param>
    /// <param name="columns">Columns in the block.</param>
    /// <exception cref="ArgumentOutOfRangeException">The block leaves the matrix.</exception>
    public MatrixView<T> Slice(int row, int column, int rows, int columns) =>
        View.Slice(row, column, rows, columns);

    /// <summary>
    /// Assert a structure without checking it, so that operations dispatch on
    /// it at compile time.
    ///
    /// This is an instance method rather than an extension so the element type
    /// comes from the receiver and only the structure has to be named:
    /// <c>a.As&lt;UpperTriangular&gt;()</c>. C# has no partial inference for
    /// explicit type arguments, so an extension would force both to be written
    /// out at every call.
    ///
    /// Unchecked is the same trust a BLAS call places in its <c>uplo</c>
    /// argument. Prefer a structured matrix that came from a factorization,
    /// where the shape holds by construction; use
    /// <see cref="StructuredMatrixExtensions.AsChecked{TStructure}"/> when the
    /// claim is about data you did not produce.
    /// </summary>
    /// <typeparam name="TStructure">The structure to assert.</typeparam>
    public StructuredMatrix<T, TStructure> As<TStructure>() where TStructure : IMatrixStructure =>
        new(View);

    /// <summary>An independent copy, packed with no stride padding.</summary>
    public Matrix<T> Clone()
    {
        var copy = new Matrix<T>(Rows, Columns);
        View.CopyTo(copy.View);
        return copy;
    }

    /// <summary>Set every element to <paramref name="value"/>.</summary>
    public void Fill(T value) => View.Fill(value);

    /// <summary>The elements in column-major order, packed with no stride padding.</summary>
    public T[] ToArray() => View.ToArray();

    /// <summary>Release the storage. Views taken from this matrix are invalid afterwards.</summary>
    public void Dispose()
    {
        if (_data is not null) { NativeMemory.AlignedFree(_data); _data = null; }
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the storage if <see cref="Dispose"/> was not called.</summary>
    ~Matrix() => Dispose();

    /// <summary>A writable view over the whole matrix.</summary>
    /// <param name="matrix">The matrix to view.</param>
    public static implicit operator MatrixView<T>(Matrix<T> matrix) => matrix.View;

    /// <summary>A read-only view over the whole matrix.</summary>
    /// <param name="matrix">The matrix to view.</param>
    public static implicit operator ReadOnlyMatrixView<T>(Matrix<T> matrix) => matrix.ReadOnlyView;
}

/// <summary>
/// Ways of making a <see cref="Matrix{T}"/>. Separate from the generic type so
/// that the element type can be inferred from the arguments instead of written
/// out at every call.
/// </summary>
public static class Matrix
{
    /// <summary>A zero matrix.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="rows">Row count.</param>
    /// <param name="columns">Column count.</param>
    /// <param name="stride">Column stride; zero means packed.</param>
    public static Matrix<T> Zeros<T>(int rows, int columns, int stride = 0)
        where T : unmanaged, INumberBase<T> => new(rows, columns, stride);

    /// <summary>A square identity matrix.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="order">Row and column count.</param>
    public static Matrix<T> Identity<T>(int order) where T : unmanaged, INumberBase<T>
    {
        var matrix = new Matrix<T>(order, order);
        for (int i = 0; i < order; i++) matrix[i, i] = T.One;
        return matrix;
    }

    /// <summary>
    /// A matrix built from values already in column-major order, which is the
    /// cheap direction: this is a straight copy.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="rows">Row count.</param>
    /// <param name="columns">Column count.</param>
    /// <param name="values">Exactly <paramref name="rows"/> * <paramref name="columns"/> elements, column by column.</param>
    /// <exception cref="ArgumentException">The element count does not match the shape.</exception>
    public static Matrix<T> FromColumnMajor<T>(int rows, int columns, ReadOnlySpan<T> values)
        where T : unmanaged, INumberBase<T>
    {
        if (values.Length != (long)rows * columns)
            throw new ArgumentException(
                $"Expected {(long)rows * columns} values for a {rows}x{columns} matrix, got {values.Length}.",
                nameof(values));

        var matrix = new Matrix<T>(rows, columns);
        for (int j = 0; j < columns; j++) values.Slice(j * rows, rows).CopyTo(matrix.Column(j));
        return matrix;
    }

    /// <summary>
    /// A matrix built from a rectangular array written the way it reads, row by
    /// row. This transposes on the way in, so it is for literals and tests
    /// rather than for bulk data.
    /// </summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="values">Values indexed as [row, column].</param>
    public static Matrix<T> FromRows<T>(T[,] values) where T : unmanaged, INumberBase<T>
    {
        ArgumentNullException.ThrowIfNull(values);

        int rows = values.GetLength(0);
        int columns = values.GetLength(1);

        var matrix = new Matrix<T>(rows, columns);
        for (int j = 0; j < columns; j++)
            for (int i = 0; i < rows; i++)
                matrix[i, j] = values[i, j];

        return matrix;
    }

    /// <summary>A copy of <paramref name="source"/>, packed with no stride padding.</summary>
    /// <typeparam name="T">Element type.</typeparam>
    /// <param name="source">The window to copy.</param>
    public static Matrix<T> From<T>(ReadOnlyMatrixView<T> source) where T : unmanaged, INumberBase<T>
    {
        var matrix = new Matrix<T>(source.Rows, source.Columns);
        for (int j = 0; j < source.Columns; j++) source.Column(j).CopyTo(matrix.Column(j));
        return matrix;
    }
}
