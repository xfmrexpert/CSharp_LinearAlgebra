using System.Numerics;
using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// An LU factorization with partial pivoting: P*A = L*U.
///
/// Unlike the primitive layer this owns its factors, so the matrix it was built
/// from is left alone. That costs a copy and is the right default for an API
/// where destroying the caller's input by surprise is the worse failure.
///
/// The factors live in a <see cref="Matrix{T}"/>, whose storage is pinned for
/// its lifetime and reclaimed by the garbage collector when the last reference
/// goes, and the pivots in an ordinary array. Nothing here holds a pointer and
/// there is nothing to dispose.
///
/// <see cref="Lower"/> and <see cref="Upper"/> are views onto the same packed
/// storage, in LAPACK's layout: L below the diagonal with an implicit unit
/// diagonal, U on and above it. Each carries a structure in its type, so a
/// solve against either dispatches to substitution at compile time and cannot
/// be pointed at the wrong triangle.
///
/// One type for every element type the library factors, <see cref="double"/>
/// and <see cref="Complex"/> today. There is no public constructor: a
/// decomposition comes only from a <c>FactorLu</c> overload, and those exist
/// only for element types with arithmetic, so a <c>LuDecomposition&lt;float&gt;</c>
/// can be named but never obtained. What differs by element type sits behind
/// the factory: a real factorization runs on the tuned kernel LU, a complex one
/// on the same blocked algorithm written generically, whose trailing updates
/// are 4M products. Members that need more than the factors -- condition
/// estimation, which needs a norm estimator for the element type -- are
/// extension methods on the closed types that have one, so asking for one that
/// does not exist fails to compile.
/// </summary>
/// <typeparam name="T">The element type.</typeparam>
public sealed class LuDecomposition<T> where T : unmanaged, INumberBase<T>
{
    private readonly Matrix<T> _factors;
    private readonly LuFactorization _factorization;
    private readonly LuSolver<T> _solver;

    internal LuDecomposition(Matrix<T> factors, LuFactorization factorization, double oneNorm, LuSolver<T> solver)
    {
        _factors = factors;
        _factorization = factorization;
        _solver = solver;
        OneNormOfA = oneNorm;
    }

    /// <summary>||A||_1 of the matrix before it was factored, for condition estimation.</summary>
    internal double OneNormOfA { get; }

    /// <summary>The factorization's pivots and diagnostics, for the tests.</summary>
    internal LuFactorization Factorization => _factorization;

    /// <summary>Rows of the factored matrix.</summary>
    public int Rows => _factorization.Rows;

    /// <summary>Columns of the factored matrix.</summary>
    public int Columns => _factorization.Columns;

    /// <summary>Whether the factored matrix was square, and so admits a solve.</summary>
    public bool IsSquare => _factorization.IsSquare;

    /// <summary>
    /// Whether an exactly zero pivot was found, which makes a solve impossible.
    ///
    /// Note that this is a narrower condition than "singular". A duplicated
    /// column is mathematically singular but its pivot comes out as rounding
    /// noise rather than an exact zero, so the factorization completes and this
    /// stays false — identical to <c>dgetrf</c>. <c>ReciprocalCondition</c>
    /// asks about numerical singularity.
    /// </summary>
    public bool IsSingular => _factorization.IsSingular;

    /// <summary>Column index of the first exactly-zero pivot, or -1 if there was none.</summary>
    public int SingularColumn => _factorization.SingularColumn;

    /// <summary>
    /// Smallest over largest modulus on U's diagonal. A cheap trouble
    /// indicator, explicitly NOT a condition number: it can be optimistic by
    /// orders of magnitude.
    /// </summary>
    public double PivotRatio => _factorization.PivotRatio;

    /// <summary>
    /// Row interchanges, zero-based. Entry k means row k was swapped with row
    /// <c>Pivots[k]</c> at step k, applied in increasing k.
    /// </summary>
    public ReadOnlySpan<int> Pivots => _factorization.Pivots;

    /// <summary>
    /// The unit lower triangular factor, as a view onto the packed storage. Its
    /// stored diagonal belongs to U and is never read.
    /// </summary>
    public StructuredMatrix<T, UnitLowerTriangular> Lower => new(_factors.View);

    /// <summary>The upper triangular factor, as a view onto the packed storage.</summary>
    public StructuredMatrix<T, UpperTriangular> Upper => new(_factors.View);

    /// <summary>Solve A*X = B, returning a fresh X.</summary>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    /// <exception cref="ArgumentException">B does not have n rows.</exception>
    public Matrix<T> Solve(ReadOnlyMatrixView<T> b)
    {
        var x = Matrix.From(b);
        SolveInPlace(x.View);
        return x;
    }

    /// <summary>Solve A*X = B in place, overwriting <paramref name="b"/> with X.</summary>
    /// <param name="b">Right-hand sides on entry, the solution on exit.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    /// <exception cref="ArgumentException">B does not have n rows.</exception>
    public void SolveInPlace(MatrixView<T> b)
    {
        RequireSolvable(b.Rows);
        _solver.Solve(_factorization, _factors.ReadOnlyView, b);
    }

    /// <summary>
    /// Solve A^H*X = B, returning a fresh X. For a real factorization A^H is
    /// A^T; for a complex one it is the conjugate transpose, which is what an
    /// adjoint operator and the norm estimator need.
    /// </summary>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    /// <exception cref="ArgumentException">B does not have n rows.</exception>
    public Matrix<T> SolveAdjoint(ReadOnlyMatrixView<T> b)
    {
        var x = Matrix.From(b);
        SolveAdjointInPlace(x.View);
        return x;
    }

    /// <summary>Solve A^H*X = B in place, overwriting <paramref name="b"/> with X.</summary>
    /// <param name="b">Right-hand sides on entry, the solution on exit.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    /// <exception cref="ArgumentException">B does not have n rows.</exception>
    public void SolveAdjointInPlace(MatrixView<T> b)
    {
        RequireSolvable(b.Rows);
        _solver.SolveAdjoint(_factorization, _factors.ReadOnlyView, b);
    }

    /// <summary>
    /// The determinant, as the product of U's diagonal with the sign of the
    /// permutation.
    ///
    /// This overflows or underflows for even moderately sized matrices — the
    /// product of n numbers has roughly n times the exponent range of one — so
    /// it is a convenience for small problems and for tests, not a numerical
    /// tool. A condition estimate is the way to ask whether a matrix is
    /// invertible; a determinant is a poor way to find out.
    /// </summary>
    /// <exception cref="InvalidOperationException">The factorization is not square.</exception>
    public T Determinant()
    {
        if (!IsSquare)
            throw new InvalidOperationException("Determinant requires a square factorization.");

        T product = T.One;
        ReadOnlyMatrixView<T> factors = _factors.ReadOnlyView;

        for (int i = 0; i < Rows; i++) product *= factors[i, i];

        ReadOnlySpan<int> pivots = Pivots;
        for (int k = 0; k < pivots.Length; k++)
            if (pivots[k] != k) product = -product;

        return product;
    }

    private void RequireSolvable(int rows)
    {
        if (!IsSquare)
            throw new InvalidOperationException("Solve requires a square factorization.");

        if (rows != Rows)
            throw new ArgumentException($"Right-hand side has {rows} rows, expected {Rows}.", nameof(rows));
    }
}

/// <summary>
/// The solves behind an <see cref="LuDecomposition{T}"/>, chosen by the factory
/// that made it. The element type is bound to its implementation there, where
/// it is known, so the generic decomposition never has to test what
/// <c>T</c> is. One virtual call per solve, against O(n^2) of work.
/// </summary>
internal abstract class LuSolver<T> where T : unmanaged, INumberBase<T>
{
    /// <summary>A*X = B in place; the shapes are already checked.</summary>
    public abstract void Solve(LuFactorization lu, ReadOnlyMatrixView<T> factors, MatrixView<T> b);

    /// <summary>A^H*X = B in place; the shapes are already checked.</summary>
    public abstract void SolveAdjoint(LuFactorization lu, ReadOnlyMatrixView<T> factors, MatrixView<T> b);
}

/// <summary>The real solves: the kernel assembly's substitution, unchanged.</summary>
internal sealed class KernelLuSolver : LuSolver<double>
{
    public static readonly KernelLuSolver Instance = new();

    private KernelLuSolver()
    {
    }

    /// <inheritdoc/>
    public override void Solve(LuFactorization lu, ReadOnlyMatrixView<double> factors, MatrixView<double> b) =>
        KernelEntry.SolveLu(lu, factors.ToOperand(), b.ToTarget());

    /// <inheritdoc/>
    public override void SolveAdjoint(LuFactorization lu, ReadOnlyMatrixView<double> factors, MatrixView<double> b) =>
        KernelEntry.SolveLuTransposed(lu, factors.ToOperand(), b.ToTarget());
}

/// <summary>The generic solves of <see cref="BlockedLu"/>, for any element type with kernels.</summary>
internal sealed class ManagedLuSolver<T, TKernels> : LuSolver<T>
    where T : unmanaged, INumberBase<T>
    where TKernels : struct, IElementKernels<T>
{
    public static readonly ManagedLuSolver<T, TKernels> Instance = new();

    private ManagedLuSolver()
    {
    }

    /// <inheritdoc/>
    public override void Solve(LuFactorization lu, ReadOnlyMatrixView<T> factors, MatrixView<T> b) =>
        BlockedLu.Solve<T, TKernels>(lu, factors, b);

    /// <inheritdoc/>
    public override void SolveAdjoint(LuFactorization lu, ReadOnlyMatrixView<T> factors, MatrixView<T> b) =>
        BlockedLu.SolveAdjoint<T, TKernels>(lu, factors, b);
}
