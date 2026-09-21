namespace Tensile.Kernels;

/// <summary>
/// A column-major operand as the kernel seam receives it: a span and the
/// shape it should be read as. This is the kernel assembly's own view type,
/// deliberately minimal, because the public view types live in the assembly
/// above and cannot be referenced from here without a cycle.
///
/// Constructing one checks the span is long enough for the shape, so the
/// only thing <see cref="KernelEntry"/> has to do is pin. The check is
/// redundant against a view that trimmed its buffer at <c>Bind</c>, and kept
/// because it is one comparison against work that is at minimum O(n^2) and it
/// is the last check before pointer arithmetic begins.
/// </summary>
internal readonly ref struct Operand
{
    /// <summary>The backing storage, at least the shape's extent long.</summary>
    public readonly ReadOnlySpan<double> Data;

    /// <summary>Rows in the operand.</summary>
    public readonly int Rows;

    /// <summary>Columns in the operand.</summary>
    public readonly int Columns;

    /// <summary>Distance in elements between consecutive column starts.</summary>
    public readonly int Stride;

    /// <summary>Bind <paramref name="data"/> to a shape.</summary>
    /// <exception cref="ArgumentException">The shape is inconsistent or the span too short.</exception>
    public Operand(ReadOnlySpan<double> data, int rows, int columns, int stride)
    {
        Shape.Check(data.Length, rows, columns, stride);

        Data = data;
        Rows = rows;
        Columns = columns;
        Stride = stride;
    }
}

/// <summary>The writable counterpart of <see cref="Operand"/>.</summary>
internal readonly ref struct Target
{
    /// <summary>The backing storage, at least the shape's extent long.</summary>
    public readonly Span<double> Data;

    /// <summary>Rows in the target.</summary>
    public readonly int Rows;

    /// <summary>Columns in the target.</summary>
    public readonly int Columns;

    /// <summary>Distance in elements between consecutive column starts.</summary>
    public readonly int Stride;

    /// <summary>Bind <paramref name="data"/> to a shape.</summary>
    /// <exception cref="ArgumentException">The shape is inconsistent or the span too short.</exception>
    public Target(Span<double> data, int rows, int columns, int stride)
    {
        Shape.Check(data.Length, rows, columns, stride);

        Data = data;
        Rows = rows;
        Columns = columns;
        Stride = stride;
    }

    /// <summary>Read-only access to the same storage.</summary>
    public Operand AsOperand() => new(Data, Rows, Columns, Stride);
}

/// <summary>The shape arithmetic behind <see cref="Operand"/> and <see cref="Target"/>, in one place.</summary>
internal static class Shape
{
    /// <summary>
    /// Elements a buffer must hold: <c>(columns - 1) * stride + rows</c>, the
    /// tight bound, computed in long so the check itself cannot wrap.
    /// </summary>
    public static void Check(int length, int rows, int columns, int stride)
    {
        if (rows < 0 || columns < 0 || stride < rows)
        {
            throw new ArgumentException(
                $"Inconsistent operand shape: {rows}x{columns} with stride {stride}.");
        }

        long extent = columns == 0 ? 0L : (long)(columns - 1) * stride + rows;

        if (extent > length)
        {
            throw new ArgumentException(
                $"Operand of {length} elements is too short for a {rows}x{columns} matrix with stride {stride}, "
                + $"which needs {extent}.");
        }
    }
}
