namespace Tensile;

/// <summary>
/// The dimensions of a column-major matrix, validated once at construction.
///
/// This type exists so that "is this shape valid" is a question answered
/// exactly once, by the only constructor, and carried as proof in the value.
/// Every piece of shape arithmetic — where an element lives, how big a buffer
/// must be, what a sub-block looks like — lives here and is computed in
/// <c>long</c> with an explicit fit check, so no size or offset can silently
/// wrap. Before this type existed the same arithmetic was re-derived by hand at
/// each call site, and one of those derivations handed back a view 16 GB past
/// its buffer.
///
/// A shape says nothing about storage. Binding it to a buffer is a separate,
/// checked step (<see cref="MatrixView{T}.Bind"/>), which is what lets the same
/// shape describe a caller's exactly-sized span, a slice of a larger buffer, or
/// a freshly allocated <see cref="Matrix{T}"/>.
/// </summary>
public readonly record struct MatrixShape
{
    /// <summary>Validate and construct. This is the only way to obtain a shape.</summary>
    /// <param name="rows">Row count, non-negative.</param>
    /// <param name="columns">Column count, non-negative.</param>
    /// <param name="stride">
    /// Distance in elements between the starts of consecutive columns. Zero
    /// means packed, i.e. equal to <paramref name="rows"/>. Otherwise at least
    /// <paramref name="rows"/>.
    /// </param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A count is negative, the stride is below the row count, or the extent
    /// the shape requires does not fit in <see cref="int"/> and so could not be
    /// addressed by a <see cref="Span{T}"/>.
    /// </exception>
    public MatrixShape(int rows, int columns, int stride = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);

        if (stride == 0) stride = rows;
        ArgumentOutOfRangeException.ThrowIfLessThan(stride, rows);

        // The last column needs no trailing padding, so the tight bound is
        // (columns - 1) * stride + rows, not stride * columns. Using the tight
        // bound lets a caller's exactly-sized buffer bind. Computed in long: the
        // product of two ints cannot overflow a long, and the comparison below
        // is what stands between an attacker-chosen dimension and the allocator.
        long extent = columns == 0 ? 0L : (long)(columns - 1) * stride + rows;

        if (extent > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(columns),
                $"A {rows}x{columns} matrix with stride {stride} needs {extent} elements, "
                + $"which exceeds the {int.MaxValue} a span can address.");
        }

        Rows = rows;
        Columns = columns;
        Stride = stride;
        RequiredExtent = (int)extent;
    }

    /// <summary>Row count.</summary>
    public int Rows { get; }

    /// <summary>Column count.</summary>
    public int Columns { get; }

    /// <summary>Distance in elements between the starts of consecutive columns. At least <see cref="Rows"/>.</summary>
    public int Stride { get; }

    /// <summary>
    /// Elements a buffer must hold for this shape: <c>(Columns - 1) * Stride + Rows</c>,
    /// or zero when there are no columns. Guaranteed to fit <see cref="int"/>.
    /// </summary>
    public int RequiredExtent { get; }

    /// <summary>Whether the shape has no elements.</summary>
    public bool IsEmpty => Rows == 0 || Columns == 0;

    /// <summary>Whether rows and columns agree.</summary>
    public bool IsSquare => Rows == Columns;

    /// <summary>Whether columns are back to back, so the whole extent is one contiguous run.</summary>
    public bool IsContiguous => Stride == Rows;

    /// <summary>
    /// Offset of element (<paramref name="row"/>, <paramref name="column"/>)
    /// from the start of the buffer.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Either index is outside the shape.</exception>
    public int OffsetOf(int row, int column)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(row, Rows);
        ArgumentOutOfRangeException.ThrowIfNegative(column);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, Columns);

        // Bounded by RequiredExtent - 1, which fits int by construction.
        return (int)((long)column * Stride + row);
    }

    /// <summary>
    /// Offset of the first element of <paramref name="column"/>. Valid for a
    /// shape with no rows, where <see cref="OffsetOf"/> would have nothing to
    /// point at.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The column is outside the shape.</exception>
    public int ColumnOffset(int column)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(column);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(column, Columns);

        // At most (Columns - 1) * Stride, which is at most RequiredExtent.
        return (int)((long)column * Stride);
    }

    /// <summary>
    /// The shape of a block within this one, sharing its stride.
    ///
    /// Checked by subtraction — <c>rows &lt;= Rows - row</c> — never as
    /// <c>row + rows &lt;= Rows</c>. The sum overflows <see cref="int"/> for
    /// large arguments, wraps negative, and passes; the subtraction cannot,
    /// because both operands are already known to be in range.
    /// </summary>
    /// <param name="row">First row of the block.</param>
    /// <param name="column">First column of the block.</param>
    /// <param name="rows">Rows in the block.</param>
    /// <param name="columns">Columns in the block.</param>
    /// <exception cref="ArgumentOutOfRangeException">The block does not lie within this shape.</exception>
    public MatrixShape Sub(int row, int column, int rows, int columns)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(row);
        ArgumentOutOfRangeException.ThrowIfNegative(column);
        ArgumentOutOfRangeException.ThrowIfNegative(rows);
        ArgumentOutOfRangeException.ThrowIfNegative(columns);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(row, Rows);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(rows, Rows - row);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(column, Columns);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(columns, Columns - column);

        // The constructor re-validates, so a sub-shape is a shape by the same
        // rule as any other. Its extent is at most this shape's, since the
        // block ends no later than the parent in either dimension.
        return new MatrixShape(rows, columns, Stride);
    }

    /// <inheritdoc/>
    public override string ToString() =>
        IsContiguous ? $"{Rows}x{Columns}" : $"{Rows}x{Columns} (stride {Stride})";
}
