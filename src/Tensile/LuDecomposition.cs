using Tensile.Primitives;

namespace Tensile;

/// <summary>
/// An LU factorization with partial pivoting: P*A = L*U.
///
/// Unlike the primitive layer this owns its factors, so the matrix it was built
/// from is left alone. That costs a copy and is the right default for an API
/// where destroying the caller's input by surprise is the worse failure; the
/// in-place route is still available through
/// <see cref="Workspace"/> and <see cref="Tensile.Primitives.Lu"/>.
///
/// <see cref="Lower"/> and <see cref="Upper"/> are views onto the same packed
/// storage, in LAPACK's layout: L below the diagonal with an implicit unit
/// diagonal, U on and above it. Each carries a structure in its type, so a
/// solve against either dispatches to substitution at compile time and cannot
/// be pointed at the wrong triangle.
/// </summary>
public sealed class LuDecomposition : IDisposable
{
    private readonly double _oneNorm;

    private Matrix<double>? _factors;
    private LuFactorization? _factorization;

    internal LuDecomposition(Matrix<double> factors, LuFactorization factorization, double oneNorm)
    {
        _factors = factors;
        _factorization = factorization;
        _oneNorm = oneNorm;
    }

    /// <summary>Rows of the factored matrix.</summary>
    public int Rows => Factorization.Rows;

    /// <summary>Columns of the factored matrix.</summary>
    public int Columns => Factorization.Columns;

    /// <summary>Whether the factored matrix was square, and so admits a solve.</summary>
    public bool IsSquare => Factorization.IsSquare;

    /// <summary>
    /// Whether an exactly zero pivot was found, which makes a solve impossible.
    ///
    /// Note that this is a narrower condition than "singular". A duplicated
    /// column is mathematically singular but its pivot comes out as rounding
    /// noise rather than an exact zero, so the factorization completes and this
    /// stays false — identical to <c>dgetrf</c>. Use
    /// <see cref="ReciprocalCondition"/> to ask about numerical singularity.
    /// </summary>
    public bool IsSingular => Factorization.IsSingular;

    /// <summary>Column index of the first exactly-zero pivot, or -1 if there was none.</summary>
    public int SingularColumn => Factorization.SingularColumn;

    /// <summary>
    /// Smallest over largest magnitude on U's diagonal. A cheap trouble
    /// indicator, explicitly NOT a condition number: it can be optimistic by
    /// orders of magnitude. <see cref="ReciprocalCondition"/> is the real
    /// answer.
    /// </summary>
    public double PivotRatio => Factorization.PivotRatio;

    /// <summary>
    /// Row interchanges, zero-based. Entry k means row k was swapped with row
    /// <c>Pivots[k]</c> at step k, applied in increasing k.
    /// </summary>
    public unsafe ReadOnlySpan<int> Pivots =>
        new(Factorization.Pivots, Math.Min(Rows, Columns));

    /// <summary>
    /// The unit lower triangular factor, as a view onto the packed storage. Its
    /// stored diagonal belongs to U and is never read.
    /// </summary>
    public StructuredMatrix<double, UnitLowerTriangular> Lower => new(Factors.View);

    /// <summary>The upper triangular factor, as a view onto the packed storage.</summary>
    public StructuredMatrix<double, UpperTriangular> Upper => new(Factors.View);

    /// <summary>
    /// Solve A*X = B, returning a fresh X.
    /// </summary>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    public Matrix<double> Solve(ReadOnlyMatrixView<double> b)
    {
        var x = Matrix.From(b);

        try
        {
            SolveInPlace(x.View);
            return x;
        }
        catch
        {
            x.Dispose();
            throw;
        }
    }

    /// <summary>Solve A*X = B in place, overwriting <paramref name="b"/> with X.</summary>
    /// <param name="b">Right-hand sides on entry, the solution on exit.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    public unsafe void SolveInPlace(MatrixView<double> b)
    {
        RequireSolvable(b.Rows);
        Lu.Solve(Factorization, b.Columns, b.Pointer, b.Stride);
    }

    /// <summary>Solve A^T*X = B, returning a fresh X.</summary>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    public Matrix<double> SolveTransposed(ReadOnlyMatrixView<double> b)
    {
        var x = Matrix.From(b);

        try
        {
            SolveTransposedInPlace(x.View);
            return x;
        }
        catch
        {
            x.Dispose();
            throw;
        }
    }

    /// <summary>Solve A^T*X = B in place, overwriting <paramref name="b"/> with X.</summary>
    /// <param name="b">Right-hand sides on entry, the solution on exit.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square, or has an exactly zero pivot.</exception>
    public unsafe void SolveTransposedInPlace(MatrixView<double> b)
    {
        RequireSolvable(b.Rows);
        Lu.SolveTransposed(Factorization, b.Columns, b.Pointer, b.Stride);
    }

    /// <summary>
    /// Estimate 1/cond_1(A), the equivalent of LAPACK's <c>dgecon</c>. Returns
    /// zero for an exactly singular factorization.
    ///
    /// The norm of the original matrix was captured before it was overwritten,
    /// so unlike <c>dgecon</c> this needs no argument and cannot be handed the
    /// wrong one.
    ///
    /// The estimator underestimates the norm of the inverse, so the result is
    /// an OVER-estimate of the reciprocal condition number: a small value
    /// reliably means ill-conditioning, a large one is weaker evidence of good
    /// conditioning.
    /// </summary>
    /// <param name="columns">Probe columns for the estimator; more costs more products.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square.</exception>
    public double ReciprocalCondition(int columns = NormEstimate.DefaultColumns)
    {
        if (!IsSquare)
            throw new InvalidOperationException("Condition estimation requires a square factorization.");

        return Condition.ReciprocalOne(_oneNorm, Factorization, columns);
    }

    /// <summary>
    /// The determinant, as the product of U's diagonal with the sign of the
    /// permutation.
    ///
    /// This overflows or underflows for even moderately sized matrices — the
    /// product of n numbers has roughly n times the exponent range of one — so
    /// it is a convenience for small problems and for tests, not a numerical
    /// tool. Ask <see cref="ReciprocalCondition"/> whether a matrix is
    /// invertible; a determinant is a poor way to find out.
    /// </summary>
    /// <exception cref="InvalidOperationException">The factorization is not square.</exception>
    public double Determinant()
    {
        if (!IsSquare)
            throw new InvalidOperationException("Determinant requires a square factorization.");

        double product = 1.0;
        MatrixView<double> factors = Factors.View;

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
            throw new ArgumentException(
                $"Right-hand side has {rows} rows, expected {Rows}.", nameof(rows));
    }

    private Matrix<double> Factors
    {
        get
        {
            ObjectDisposedException.ThrowIf(_factors is null, this);
            return _factors!;
        }
    }

    private LuFactorization Factorization
    {
        get
        {
            ObjectDisposedException.ThrowIf(_factorization is null, this);
            return _factorization!;
        }
    }

    /// <summary>Release the factors and the pivot array.</summary>
    public void Dispose()
    {
        _factorization?.Dispose();
        _factorization = null;

        _factors?.Dispose();
        _factors = null;

        GC.SuppressFinalize(this);
    }
}
