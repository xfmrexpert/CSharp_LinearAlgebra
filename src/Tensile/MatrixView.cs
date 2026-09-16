using System.Runtime.CompilerServices;

namespace Tensile;

/// <summary>
/// A borrowed, stride-aware window onto column-major storage.
///
/// This is the currency of the operation layer: everything that reads or writes
/// a matrix takes a view, so a submatrix costs nothing and no operation needs an
/// overload for "the whole thing" versus "part of it".
///
/// It is a <c>ref struct</c> deliberately. A view does not own its storage and
/// is only valid while the <see cref="Matrix{T}"/> or pinned buffer behind it
/// is alive, so the compiler is enlisted to stop it being captured in a field,
/// boxed, or smuggled into an async method — the three ways a borrowed pointer
/// usually escapes.
///
/// Layout is column-major with unit row stride, matching LAPACK and the
/// micro-kernels: element (i, j) sits at <c>origin[j * Stride + i]</c>, and
/// <see cref="Stride"/> may exceed <see cref="Rows"/> so that a submatrix of a
/// larger buffer is a view rather than a copy.
/// </summary>
/// <typeparam name="T">Element type. Unmanaged so the storage can be native and aligned.</typeparam>
public readonly unsafe ref struct MatrixView<T> where T : unmanaged
{
    private readonly T* _origin;

    /// <summary>Rows in the window.</summary>
    public int Rows { get; }

    /// <summary>Columns in the window.</summary>
    public int Columns { get; }

    /// <summary>
    /// Distance in elements between the starts of consecutive columns. At least
    /// <see cref="Rows"/>; larger when this window sits inside a wider buffer.
    /// </summary>
    public int Stride { get; }

    /// <summary>Wrap raw column-major storage. The caller guarantees the extent and lifetime.</summary>
    /// <param name="origin">Address of element (0, 0).</param>
    /// <param name="rows">Rows in the window.</param>
    /// <param name="columns">Columns in the window.</param>
    /// <param name="stride">Column stride, at least <paramref name="rows"/>.</param>
    public MatrixView(T* origin, int rows, int columns, int stride)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        ArgumentOutOfRangeException.ThrowIfLessThan(stride, rows);

        _origin = origin;
        Rows = rows;
        Columns = columns;
        Stride = stride;
    }

    /// <summary>Address of element (0, 0). For handing to the primitive layer.</summary>
    internal T* Pointer => _origin;

    /// <summary>Whether the window has no elements.</summary>
    public bool IsEmpty => Rows == 0 || Columns == 0;

    /// <summary>
    /// Whether the columns are back to back, so the whole window is one
    /// contiguous run and can be copied or cleared in a single operation.
    /// </summary>
    public bool IsContiguous => Stride == Rows;

    /// <summary>Element (<paramref name="row"/>, <paramref name="column"/>), by reference.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either index is outside the window.</exception>
    public ref T this[int row, int column]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfNegative(row);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Rows);
            ArgumentOutOfRangeException.ThrowIfNegative(column);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, Columns);

            return ref _origin[(nint)column * Stride + row];
        }
    }

    /// <summary>
    /// One column as a span. Free, because a column is contiguous in
    /// column-major storage — the reason there is no matching <c>Row</c>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The column index is outside the window.</exception>
    public Span<T> Column(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Columns);

        return new Span<T>(_origin + (nint)index * Stride, Rows);
    }

    /// <summary>A window onto part of this window. Shares storage; copies nothing.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The requested block leaves this window.</exception>
    public MatrixView<T> Slice(int row, int column, int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfNegative(column);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(row + rows, Rows);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(column + columns, Columns);

        return new MatrixView<T>(_origin + (nint)column * Stride + row, rows, columns, Stride);
    }

    /// <summary>Copy every element of this window into <paramref name="destination"/>.</summary>
    /// <exception cref="ArgumentException">The shapes differ.</exception>
    public void CopyTo(MatrixView<T> destination)
    {
        if (destination.Rows != Rows || destination.Columns != Columns)
            throw new ArgumentException(
                $"Shape mismatch: source is {Rows}x{Columns}, destination is {destination.Rows}x{destination.Columns}.",
                nameof(destination));

        for (int j = 0; j < Columns; j++) Column(j).CopyTo(destination.Column(j));
    }

    /// <summary>Set every element of this window to <paramref name="value"/>.</summary>
    public void Fill(T value)
    {
        for (int j = 0; j < Columns; j++) Column(j).Fill(value);
    }

    /// <summary>Copy the window into a fresh column-major array, packed with no stride padding.</summary>
    public T[] ToArray()
    {
        var result = new T[(long)Rows * Columns];

        for (int j = 0; j < Columns; j++)
            Column(j).CopyTo(result.AsSpan(j * Rows, Rows));

        return result;
    }

    /// <summary>Narrow to a read-only window over the same storage.</summary>
    public static implicit operator ReadOnlyMatrixView<T>(MatrixView<T> view) =>
        new(view._origin, view.Rows, view.Columns, view.Stride);
}

/// <summary>
/// A borrowed window that cannot be written through. Operands that an operation
/// only reads are typed this way, so "which arguments does this overwrite" is
/// answered by the signature rather than by the documentation.
/// </summary>
/// <typeparam name="T">Element type.</typeparam>
public readonly unsafe ref struct ReadOnlyMatrixView<T> where T : unmanaged
{
    private readonly T* _origin;

    /// <summary>Rows in the window.</summary>
    public int Rows { get; }

    /// <summary>Columns in the window.</summary>
    public int Columns { get; }

    /// <summary>Distance in elements between the starts of consecutive columns.</summary>
    public int Stride { get; }

    /// <summary>Wrap raw column-major storage. The caller guarantees the extent and lifetime.</summary>
    /// <param name="origin">Address of element (0, 0).</param>
    /// <param name="rows">Rows in the window.</param>
    /// <param name="columns">Columns in the window.</param>
    /// <param name="stride">Column stride, at least <paramref name="rows"/>.</param>
    public ReadOnlyMatrixView(T* origin, int rows, int columns, int stride)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        ArgumentOutOfRangeException.ThrowIfLessThan(stride, rows);

        _origin = origin;
        Rows = rows;
        Columns = columns;
        Stride = stride;
    }

    /// <summary>Address of element (0, 0). For handing to the primitive layer.</summary>
    internal T* Pointer => _origin;

    /// <summary>Whether the window has no elements.</summary>
    public bool IsEmpty => Rows == 0 || Columns == 0;

    /// <summary>Element (<paramref name="row"/>, <paramref name="column"/>), by read-only reference.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either index is outside the window.</exception>
    public ref readonly T this[int row, int column]
    {
        get
        {
            ArgumentOutOfRangeException.ThrowIfNegative(row);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Rows);
            ArgumentOutOfRangeException.ThrowIfNegative(column);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, Columns);

            return ref _origin[(nint)column * Stride + row];
        }
    }

    /// <summary>One column as a read-only span.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The column index is outside the window.</exception>
    public ReadOnlySpan<T> Column(int index)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(index);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, Columns);

        return new ReadOnlySpan<T>(_origin + (nint)index * Stride, Rows);
    }

    /// <summary>A read-only window onto part of this window. Shares storage; copies nothing.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The requested block leaves this window.</exception>
    public ReadOnlyMatrixView<T> Slice(int row, int column, int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfNegative(column);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(row + rows, Rows);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(column + columns, Columns);

        return new ReadOnlyMatrixView<T>(_origin + (nint)column * Stride + row, rows, columns, Stride);
    }

    /// <summary>Copy the window into a fresh column-major array, packed with no stride padding.</summary>
    public T[] ToArray()
    {
        var result = new T[(long)Rows * Columns];

        for (int j = 0; j < Columns; j++)
            Column(j).CopyTo(result.AsSpan(j * Rows, Rows));

        return result;
    }
}
