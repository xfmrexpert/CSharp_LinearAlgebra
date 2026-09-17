using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// A claim about a matrix's shape, carried in the type system rather than in a
/// function name.
///
/// LAPACK encodes this distinction in its naming — <c>dgesv</c> against
/// <c>dtrsv</c> against <c>dposv</c> — so choosing the wrong one compiles
/// cleanly and returns a confidently wrong answer. Making the structure a type
/// parameter moves that decision to the compiler: a
/// <see cref="StructuredMatrix{T, TStructure}"/> can only be passed to an
/// operation that accepts its structure, and the operation is selected at
/// compile time through static abstract dispatch, so the safety costs nothing
/// at run time.
///
/// A structure says which part of the storage an operation reads. It does not
/// say the rest is zero, and it deliberately cannot: LU packs L and U into one
/// array, so the triangle a structure ignores routinely holds the other factor.
/// See <see cref="UnreferencedPartIsZero"/> for the stronger property, which is
/// a question you ask explicitly rather than something the type asserts.
/// </summary>
public interface IMatrixStructure
{
    /// <summary>Short name, for diagnostics and error messages.</summary>
    static abstract string Name { get; }

    /// <summary>
    /// Whether the elements this structure does not reference are all exactly
    /// zero — that is, whether the matrix stands alone in this shape rather
    /// than sharing storage with another factor.
    ///
    /// False for the triangles of a packed LU, and that is not a fault: the
    /// solves never read the other triangle. Use this to validate a matrix you
    /// believe is genuinely triangular, not to check a factorization.
    /// </summary>
    /// <param name="a">The matrix to inspect.</param>
    static abstract bool UnreferencedPartIsZero(ReadOnlyMatrixView<double> a);
}

/// <summary>
/// A structure that can be solved against directly, by substitution, with no
/// factorization first. This is the constraint that makes
/// <c>A.Solve(b)</c> pick back substitution over LU without a run-time branch.
/// </summary>
public interface ITriangularStructure : IMatrixStructure
{
    /// <summary>Solve A * X = B in place, B holding the right-hand sides on entry and X on exit.</summary>
    /// <param name="a">The triangular operand, order n.</param>
    /// <param name="b">n x nrhs right-hand sides, overwritten with the solution.</param>
    static abstract void SolveInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b);

    /// <summary>Solve A^T * X = B in place.</summary>
    /// <param name="a">The triangular operand, order n.</param>
    /// <param name="b">n x nrhs right-hand sides, overwritten with the solution.</param>
    static abstract void SolveTransposedInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b);
}

/// <summary>
/// No assumed shape. Every element is referenced, and a solve needs a
/// factorization.
/// </summary>
public readonly struct General : IMatrixStructure
{
    /// <inheritdoc/>
    public static string Name => "general";

    /// <summary>Vacuously true: a general matrix has no unreferenced part.</summary>
    /// <param name="a">The matrix to inspect.</param>
    public static bool UnreferencedPartIsZero(ReadOnlyMatrixView<double> a) => true;
}

/// <summary>
/// Upper triangular with an explicit diagonal. Only the diagonal and above are
/// referenced. This is the shape of U in an LU factorization.
/// </summary>
public readonly struct UpperTriangular : ITriangularStructure
{
    /// <inheritdoc/>
    public static string Name => "upper triangular";

    /// <inheritdoc/>
    public static bool UnreferencedPartIsZero(ReadOnlyMatrixView<double> a)
    {
        for (int j = 0; j < a.Columns; j++)
            for (int i = j + 1; i < a.Rows; i++)
                if (a[i, j] != 0.0) return false;

        return true;
    }

    /// <inheritdoc/>
    public static void SolveInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b) =>
        KernelEntry.SolveUpper(a.ToOperand(), b.ToTarget());

    /// <inheritdoc/>
    public static void SolveTransposedInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b) =>
        KernelEntry.SolveUpperTransposed(a.ToOperand(), b.ToTarget());
}

/// <summary>
/// Lower triangular with an explicit diagonal. Only the diagonal and below are
/// referenced.
/// </summary>
public readonly struct LowerTriangular : ITriangularStructure
{
    /// <inheritdoc/>
    public static string Name => "lower triangular";

    /// <inheritdoc/>
    public static bool UnreferencedPartIsZero(ReadOnlyMatrixView<double> a)
    {
        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < Math.Min(j, a.Rows); i++)
                if (a[i, j] != 0.0) return false;

        return true;
    }

    /// <inheritdoc/>
    public static void SolveInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b) =>
        KernelEntry.SolveLower(a.ToOperand(), b.ToTarget());

    /// <inheritdoc/>
    public static void SolveTransposedInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b) =>
        KernelEntry.SolveLowerTransposed(a.ToOperand(), b.ToTarget());
}

/// <summary>
/// Lower triangular with an implicit unit diagonal: the stored diagonal is
/// never read and is assumed to be 1. This is the shape of L in an LU
/// factorization, where the diagonal slot holds U's diagonal instead.
/// </summary>
public readonly struct UnitLowerTriangular : ITriangularStructure
{
    /// <inheritdoc/>
    public static string Name => "unit lower triangular";

    /// <summary>
    /// Whether everything strictly above the diagonal is zero. The diagonal
    /// itself is not checked, because it is not referenced — in packed LU
    /// storage it belongs to U.
    /// </summary>
    /// <param name="a">The matrix to inspect.</param>
    public static bool UnreferencedPartIsZero(ReadOnlyMatrixView<double> a)
    {
        for (int j = 0; j < a.Columns; j++)
            for (int i = 0; i < Math.Min(j, a.Rows); i++)
                if (a[i, j] != 0.0) return false;

        return true;
    }

    /// <inheritdoc/>
    public static void SolveInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b) =>
        KernelEntry.SolveLowerUnit(a.ToOperand(), b.ToTarget());

    /// <inheritdoc/>
    public static void SolveTransposedInPlace(ReadOnlyMatrixView<double> a, MatrixView<double> b) =>
        KernelEntry.SolveLowerUnitTransposed(a.ToOperand(), b.ToTarget());
}

/// <summary>
/// A window onto a matrix together with a compile-time claim about its shape.
///
/// Borrowed rather than owning, and a <c>ref struct</c> for the same reason
/// <see cref="MatrixView{T}"/> is: the claim is only meaningful while the
/// storage behind it is alive.
/// </summary>
/// <typeparam name="T">Element type.</typeparam>
/// <typeparam name="TStructure">The asserted structure.</typeparam>
public readonly ref struct StructuredMatrix<T, TStructure>
    where T : unmanaged
    where TStructure : IMatrixStructure
{
    /// <summary>Wrap a view with a structure claim.</summary>
    /// <param name="view">The underlying window.</param>
    public StructuredMatrix(MatrixView<T> view) => View = view;

    /// <summary>The underlying window.</summary>
    public MatrixView<T> View { get; }

    /// <summary>Rows in the window.</summary>
    public int Rows => View.Rows;

    /// <summary>Columns in the window.</summary>
    public int Columns => View.Columns;

    /// <summary>Drop the structure claim.</summary>
    /// <param name="matrix">The structured matrix.</param>
    public static implicit operator MatrixView<T>(StructuredMatrix<T, TStructure> matrix) => matrix.View;
}

/// <summary>Checking structure claims that were not established by construction.</summary>
public static class StructuredMatrixExtensions
{
    /// <summary>
    /// Assert a structure, verifying first that the unreferenced part really is
    /// zero. O(n^2), so this is for validating input rather than for inner
    /// loops, and it rejects a packed factorization by design.
    ///
    /// Only the structure is a type parameter. The element type is fixed at
    /// <see cref="double"/> because that is what the verification reads; an
    /// unused type parameter here would let a caller write
    /// <c>AsChecked&lt;float, UpperTriangular&gt;()</c> on a matrix of doubles
    /// and misdescribe what was checked.
    /// </summary>
    /// <typeparam name="TStructure">The structure to assert.</typeparam>
    /// <param name="matrix">The matrix to reinterpret.</param>
    /// <exception cref="ArgumentException">The matrix does not have the claimed shape.</exception>
    public static StructuredMatrix<double, TStructure> AsChecked<TStructure>(this Matrix<double> matrix)
        where TStructure : IMatrixStructure
    {
        ArgumentNullException.ThrowIfNull(matrix);

        if (!TStructure.UnreferencedPartIsZero(matrix.ReadOnlyView))
            throw new ArgumentException(
                $"Matrix is not {TStructure.Name}: elements outside the referenced triangle are non-zero.",
                nameof(matrix));

        return new StructuredMatrix<double, TStructure>(matrix.View);
    }
}
