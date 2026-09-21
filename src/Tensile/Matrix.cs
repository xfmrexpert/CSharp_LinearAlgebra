using System.Numerics;
using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// A dense column-major matrix that owns its storage.
///
/// Storage is a managed array on the pinned object heap, obtained with
/// <c>GC.AllocateArray(..., pinned: true)</c>, and padded so that the first
/// element sits on a 64-byte boundary. That choice does three things at once:
///
/// - The array never moves, so a view of it can be handed to a kernel without
///   further pinning and the aligned start stays aligned.
/// - A <see cref="Span{T}"/> over a managed array is a reference the garbage
///   collector tracks. While any <see cref="MatrixView{T}"/> of this matrix is
///   live on any thread's stack, the storage cannot be reclaimed. Use-after-free
///   is not prevented here; it is unexpressible.
/// - There is nothing to dispose. No <c>IDisposable</c>, no finalizer, no
///   double-free, no leak when a caller forgets. The storage is reclaimed when
///   the last reference to it is gone, like any other array.
///
/// The type is generic so that storage, views, slicing and the structure
/// vocabulary are written once. Arithmetic is currently supplied only for
/// <see cref="double"/>, through extension methods on the closed type, which is
/// why <c>Matrix&lt;float&gt;</c> compiles and holds data but has nothing to
/// multiply it with yet.
///
/// A pinned array is never compacted, so a hot loop that creates thousands of
/// tiny matrices will fragment the heap. For a solver holding a handful of large
/// operands this is the behaviour you want; for churn, reuse storage through a
/// <see cref="Workspace"/> and views instead.
/// </summary>
/// <typeparam name="T">Element type.</typeparam>
public sealed class Matrix<T> where T : unmanaged, INumberBase<T>
{
    private readonly T[] _storage;
    private readonly int _offset;

    /// <summary>Allocate a zeroed matrix.</summary>
    /// <param name="rows">Row count.</param>
    /// <param name="columns">Column count.</param>
    /// <param name="stride">Column stride; zero means packed.</param>
    /// <exception cref="ArgumentOutOfRangeException">The shape is invalid or too large to own. See <see cref="MatrixShape"/>.</exception>
    public Matrix(int rows, int columns, int stride = 0)
        : this(new MatrixShape(rows, columns, stride))
    {
    }

    /// <summary>Allocate a zeroed matrix of the given shape.</summary>
    /// <param name="shape">The shape, already validated.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The shape is valid but too large to own: the extent plus alignment
    /// padding exceeds <see cref="Array.MaxLength"/>. A shape that large can
    /// still be bound to a caller's own span.
    /// </exception>
    /// <exception cref="AllocationLimitException">The extent exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    public Matrix(MatrixShape shape)
    {
        int padding = Alignment.PaddingElements<T>();

        // MatrixShape guarantees the extent fits int, which is the span limit.
        // An array has a slightly lower limit, and we need room to align, so
        // the owning type applies the stricter check. Computed in long; this is
        // the last integer comparison before the allocator, and it must not be
        // the one that wraps.
        if ((long)shape.RequiredExtent + padding > Array.MaxLength)
        {
            throw new ArgumentOutOfRangeException(
                nameof(shape),
                $"A {shape} matrix needs {shape.RequiredExtent} elements plus {padding} for alignment, "
                + $"which exceeds the {Array.MaxLength} an array can hold. Bind a caller-owned span instead.");
        }

        Shape = shape;
        _storage = Storage.Pinned<T>(shape.RequiredExtent, padding, $"a {shape} matrix");
        _offset = Alignment.AlignedOffset(_storage);
    }

    /// <summary>The dimensions of this matrix.</summary>
    public MatrixShape Shape { get; }

    /// <summary>Row count.</summary>
    public int Rows => Shape.Rows;

    /// <summary>Column count.</summary>
    public int Columns => Shape.Columns;

    /// <summary>Distance in elements between the starts of consecutive columns.</summary>
    public int Stride => Shape.Stride;

    /// <summary>Whether the matrix has no elements.</summary>
    public bool IsEmpty => Shape.IsEmpty;

    /// <summary>Whether the matrix is square, and so a candidate for solving and factorization.</summary>
    public bool IsSquare => Shape.IsSquare;

    /// <summary>Element (<paramref name="row"/>, <paramref name="column"/>), by reference.</summary>
    /// <exception cref="ArgumentOutOfRangeException">Either index is out of range.</exception>
    public ref T this[int row, int column] => ref View[row, column];

    /// <summary>A writable view over the whole matrix.</summary>
    public MatrixView<T> View =>
        MatrixView<T>.Bind(_storage.AsSpan(_offset, Shape.RequiredExtent), Shape);

    /// <summary>A read-only view over the whole matrix.</summary>
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
    /// An instance method rather than an extension so the element type comes
    /// from the receiver and only the structure has to be named:
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

    /// <summary>A writable view over the whole matrix; the same as <see cref="View"/>.</summary>
    /// <param name="matrix">The matrix to view.</param>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    public static implicit operator MatrixView<T>(Matrix<T> matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        return matrix.View;
    }

    /// <summary>A read-only view over the whole matrix; the same as <see cref="ReadOnlyView"/>.</summary>
    /// <param name="matrix">The matrix to view.</param>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    public static implicit operator ReadOnlyMatrixView<T>(Matrix<T> matrix)
    {
        ArgumentNullException.ThrowIfNull(matrix);
        return matrix.ReadOnlyView;
    }
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
        // Validate the shape first, so a hostile dimension is reported as such
        // rather than as a count mismatch against a product that wrapped.
        var shape = new MatrixShape(rows, columns);

        if (values.Length != (long)rows * columns)
        {
            throw new ArgumentException(
                $"Expected {(long)rows * columns} values for a {rows}x{columns} matrix, got {values.Length}.",
                nameof(values));
        }

        var matrix = new Matrix<T>(shape);

        // An empty matrix has nothing to copy, and a 0 x 2^31 one has two
        // billion columns to not copy it into. Every column walk in the
        // library short-circuits on IsEmpty for this reason; the fuzzer found
        // the case as a 36-second hang.
        if (shape.IsEmpty) return matrix;

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
        if (source.IsEmpty) return matrix;

        for (int j = 0; j < source.Columns; j++) source.Column(j).CopyTo(matrix.Column(j));
        return matrix;
    }
}
