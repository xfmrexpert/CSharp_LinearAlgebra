namespace Tensile;

/// <summary>
/// A borrowed, stride-aware window onto column-major storage, backed by a
/// <see cref="Span{T}"/>.
///
/// This is the currency of the operation layer: everything that reads or writes
/// a matrix takes a view, so a submatrix costs nothing and no operation needs an
/// overload for "the whole thing" versus "part of it".
///
/// Two properties are the point of the design and are worth stating plainly.
///
/// First, there is no way to construct a view from a raw pointer. A view is
/// obtained by <see cref="Bind"/>ing a span to a shape — which checks that the
/// span is long enough — or by slicing another view. A caller who has a pointer
/// writes <c>new Span&lt;T&gt;(p, length)</c> themselves: that is their unsafe
/// act, correctly attributed, and it forces them to state the length, which is
/// exactly the fact this type needs.
///
/// Second, the buffer's extent travels with the view, so the runtime is the last
/// line of defence. Every column and sub-block is a <c>Span.Slice</c>, which the
/// BCL bounds-checks. If our own offset arithmetic were ever wrong, the result
/// would be an exception rather than a pointer outside the buffer. That is the
/// difference between a bug and a vulnerability.
///
/// It is a <c>ref struct</c> because a span is, and because a borrowed window
/// must not be captured in a field, boxed, or smuggled into an async method.
/// </summary>
/// <typeparam name="T">Element type. Unmanaged so that storage can be pinned and handed to the kernels.</typeparam>
public readonly ref struct MatrixView<T> where T : unmanaged
{
    // Exactly Shape.RequiredExtent long: Bind trims the incoming span, so the
    // view's footprint and its buffer coincide and Overlaps is exact.
    private readonly Span<T> _buffer;

    private MatrixView(Span<T> buffer, MatrixShape shape)
    {
        _buffer = buffer;
        Shape = shape;
    }

    /// <summary>
    /// View <paramref name="buffer"/> as a matrix of the given shape.
    /// </summary>
    /// <param name="buffer">Column-major storage, at least <see cref="MatrixShape.RequiredExtent"/> long.</param>
    /// <param name="shape">The shape to impose.</param>
    /// <exception cref="ArgumentException">The buffer is too short for the shape.</exception>
    public static MatrixView<T> Bind(Span<T> buffer, MatrixShape shape)
    {
        if (buffer.Length < shape.RequiredExtent)
        {
            throw new ArgumentException(
                $"Buffer of {buffer.Length} elements is too short for a {shape} matrix, which needs {shape.RequiredExtent}.",
                nameof(buffer));
        }

        return new MatrixView<T>(buffer.Slice(0, shape.RequiredExtent), shape);
    }

    /// <summary>The dimensions of this window.</summary>
    public MatrixShape Shape { get; }

    /// <summary>Rows in the window.</summary>
    public int Rows => Shape.Rows;

    /// <summary>Columns in the window.</summary>
    public int Columns => Shape.Columns;

    /// <summary>Distance in elements between the starts of consecutive columns.</summary>
    public int Stride => Shape.Stride;

    /// <summary>Whether the window has no elements.</summary>
    public bool IsEmpty => Shape.IsEmpty;

    /// <summary>Whether the columns are back to back, so the window is one contiguous run.</summary>
    public bool IsContiguous => Shape.IsContiguous;

    /// <summary>
    /// The backing span, exactly <see cref="MatrixShape.RequiredExtent"/> long.
    /// For the kernel entry seam, which pins it for the duration of a call.
    /// </summary>
    internal Span<T> Buffer => _buffer;

    /// <summary>Element (<paramref name="row"/>, <paramref name="column"/>), by reference.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either index is outside the window.</exception>
    public ref T this[int row, int column] => ref _buffer[Shape.OffsetOf(row, column)];

    /// <summary>
    /// One column as a span. Free, because a column is contiguous in
    /// column-major storage — the reason there is no matching <c>Row</c>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The column index is outside the window.</exception>
    public Span<T> Column(int index) => _buffer.Slice(Shape.ColumnOffset(index), Shape.Rows);

    /// <summary>A window onto part of this window. Shares storage; copies nothing.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The requested block leaves this window.</exception>
    public MatrixView<T> Slice(int row, int column, int rows, int columns)
    {
        MatrixShape sub = Shape.Sub(row, column, rows, columns);

        // An empty block may legitimately sit at the far edge, where OffsetOf
        // would have no element to name; it needs no storage either way.
        if (sub.IsEmpty) return new MatrixView<T>(_buffer.Slice(0, 0), sub);

        // Defended twice: Sub checked the block lies within this shape, and
        // Span.Slice checks the derived extent lies within the buffer.
        return new MatrixView<T>(_buffer.Slice(Shape.OffsetOf(row, column), sub.RequiredExtent), sub);
    }

    /// <summary>Copy every element of this window into <paramref name="destination"/>.</summary>
    /// <exception cref="ArgumentException">The shapes differ.</exception>
    /// <exception cref="AllocationLimitException">The windows overlap and a staging copy would exceed <see cref="TensileLimits.MaxElements"/>.</exception>
    public void CopyTo(MatrixView<T> destination)
    {
        if (destination.Rows != Rows || destination.Columns != Columns)
        {
            throw new ArgumentException(
                $"Shape mismatch: source is {Rows}x{Columns}, destination is {destination.Rows}x{destination.Columns}.",
                nameof(destination));
        }

        // Nothing to copy, and possibly billions of columns to not copy.
        if (IsEmpty) return;

        // Column by column is not overlap-safe on its own: Span.CopyTo protects
        // each column, but if the windows overlap across columns, writing
        // destination column j can destroy source column j+1 before it is
        // read. Staging costs an allocation only in that case and is correct
        // whatever the two strides are.
        if (Overlaps(destination))
        {
            T[] staged = ToArray();

            for (int j = 0; j < Columns; j++)
                staged.AsSpan(j * Rows, Rows).CopyTo(destination.Column(j));

            return;
        }

        for (int j = 0; j < Columns; j++) Column(j).CopyTo(destination.Column(j));
    }

    /// <summary>
    /// Whether this window and <paramref name="other"/> share any storage. Since
    /// each buffer is trimmed to its footprint at <see cref="Bind"/>, this is an
    /// exact address-range test, not a conservative one.
    /// </summary>
    public bool Overlaps(MatrixView<T> other) => ((ReadOnlySpan<T>)_buffer).Overlaps(other._buffer);

    /// <summary>Set every element of this window to <paramref name="value"/>.</summary>
    public void Fill(T value)
    {
        if (IsEmpty) return;

        for (int j = 0; j < Columns; j++) Column(j).Fill(value);
    }

    /// <summary>Copy the window into a fresh column-major array, packed with no stride padding.</summary>
    /// <exception cref="AllocationLimitException">The packed size exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    public T[] ToArray()
    {
        // Rows * Columns is at most RequiredExtent, which fits int by I2.
        T[] result = Storage.Array<T>(Rows * Columns, $"a packed copy of a {Rows}x{Columns} window");

        if (IsEmpty) return result;

        for (int j = 0; j < Columns; j++)
            Column(j).CopyTo(result.AsSpan(j * Rows, Rows));

        return result;
    }

    /// <summary>Narrow to a read-only window over the same storage.</summary>
    public static implicit operator ReadOnlyMatrixView<T>(MatrixView<T> view) =>
        new(view._buffer, view.Shape);
}

/// <summary>
/// A borrowed window that cannot be written through. Operands an operation only
/// reads are typed this way, so "which arguments does this overwrite" is
/// answered by the signature rather than by the documentation.
/// </summary>
/// <typeparam name="T">Element type.</typeparam>
public readonly ref struct ReadOnlyMatrixView<T> where T : unmanaged
{
    private readonly ReadOnlySpan<T> _buffer;

    internal ReadOnlyMatrixView(ReadOnlySpan<T> buffer, MatrixShape shape)
    {
        _buffer = buffer;
        Shape = shape;
    }

    /// <summary>View <paramref name="buffer"/> as a read-only matrix of the given shape.</summary>
    /// <param name="buffer">Column-major storage, at least <see cref="MatrixShape.RequiredExtent"/> long.</param>
    /// <param name="shape">The shape to impose.</param>
    /// <exception cref="ArgumentException">The buffer is too short for the shape.</exception>
    public static ReadOnlyMatrixView<T> Bind(ReadOnlySpan<T> buffer, MatrixShape shape)
    {
        if (buffer.Length < shape.RequiredExtent)
        {
            throw new ArgumentException(
                $"Buffer of {buffer.Length} elements is too short for a {shape} matrix, which needs {shape.RequiredExtent}.",
                nameof(buffer));
        }

        return new ReadOnlyMatrixView<T>(buffer.Slice(0, shape.RequiredExtent), shape);
    }

    /// <summary>The dimensions of this window.</summary>
    public MatrixShape Shape { get; }

    /// <summary>Rows in the window.</summary>
    public int Rows => Shape.Rows;

    /// <summary>Columns in the window.</summary>
    public int Columns => Shape.Columns;

    /// <summary>Distance in elements between the starts of consecutive columns.</summary>
    public int Stride => Shape.Stride;

    /// <summary>Whether the window has no elements.</summary>
    public bool IsEmpty => Shape.IsEmpty;

    /// <summary>The backing span, for the kernel entry seam.</summary>
    internal ReadOnlySpan<T> Buffer => _buffer;

    /// <summary>Element (<paramref name="row"/>, <paramref name="column"/>), by read-only reference.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either index is outside the window.</exception>
    public ref readonly T this[int row, int column] => ref _buffer[Shape.OffsetOf(row, column)];

    /// <summary>One column as a read-only span.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The column index is outside the window.</exception>
    public ReadOnlySpan<T> Column(int index) => _buffer.Slice(Shape.ColumnOffset(index), Shape.Rows);

    /// <summary>A read-only window onto part of this window. Shares storage; copies nothing.</summary>
    /// <exception cref="ArgumentOutOfRangeException">The requested block leaves this window.</exception>
    public ReadOnlyMatrixView<T> Slice(int row, int column, int rows, int columns)
    {
        MatrixShape sub = Shape.Sub(row, column, rows, columns);

        if (sub.IsEmpty) return new ReadOnlyMatrixView<T>(_buffer.Slice(0, 0), sub);

        return new ReadOnlyMatrixView<T>(_buffer.Slice(Shape.OffsetOf(row, column), sub.RequiredExtent), sub);
    }

    /// <summary>Copy the window into a fresh column-major array, packed with no stride padding.</summary>
    /// <exception cref="AllocationLimitException">The packed size exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    public T[] ToArray()
    {
        // Rows * Columns is at most RequiredExtent, which fits int by I2.
        T[] result = Storage.Array<T>(Rows * Columns, $"a packed copy of a {Rows}x{Columns} window");

        if (IsEmpty) return result;

        for (int j = 0; j < Columns; j++)
            Column(j).CopyTo(result.AsSpan(j * Rows, Rows));

        return result;
    }
}
