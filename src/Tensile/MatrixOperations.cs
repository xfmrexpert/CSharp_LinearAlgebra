using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// The fluent surface: operations on <see cref="Matrix{T}"/> that allocate
/// their result.
///
/// Every method here has a counterpart on <see cref="Workspace"/> or an
/// <c>...InPlace</c> form that writes into storage the caller already owns.
/// This layer is the convenient one, not the efficient one — a loop that calls
/// <see cref="Multiply(Matrix{double}, ReadOnlyMatrixView{double}, Workspace)"/>
/// allocates a matrix per iteration.
///
/// Arithmetic is supplied for <see cref="double"/>, and for
/// <see cref="System.Numerics.Complex"/> where it exists so far (products,
/// in <c>MatrixOperations.Complex.cs</c>). The extension targets are closed
/// types -- <c>Matrix&lt;double&gt;</c>, <c>Matrix&lt;Complex&gt;</c> --
/// rather than an open <c>Matrix&lt;T&gt;</c>, so adding a numeric type adds
/// overloads without changing any signature here, and a type with no
/// arithmetic fails to compile rather than at run time.
///
/// No method here touches a pointer, and none could: this assembly compiles
/// with unsafe code disallowed. Anything that needs one hands a view to the
/// kernel assembly's entry seam, which pins it for exactly the duration of a
/// call.
/// </summary>
public static partial class MatrixOperations
{
    /// <summary>A*B, as a new matrix.</summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <exception cref="ArgumentException">The inner dimensions disagree.</exception>
    public static Matrix<double> Multiply(
        this Matrix<double> a, ReadOnlyMatrixView<double> b, Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        var result = new Matrix<double>(a.Rows, b.Columns);
        (workspace ?? Workspace.Shared).Multiply(a.ReadOnlyView, b, result.View);
        return result;
    }

    /// <summary>
    /// destination := beta*destination + alpha*A*B, writing into storage the
    /// caller owns.
    /// </summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="destination">Destination, m x n.</param>
    /// <param name="alpha">Scalar on the product.</param>
    /// <param name="beta">Scalar on the existing contents of the destination.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <exception cref="ArgumentException">The shapes are not conformable.</exception>
    public static void MultiplyInto(
        this Matrix<double> a,
        ReadOnlyMatrixView<double> b,
        MatrixView<double> destination,
        double alpha = 1.0,
        double beta = 0.0,
        Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        (workspace ?? Workspace.Shared).Multiply(a.ReadOnlyView, b, destination, alpha, beta);
    }

    /// <summary>
    /// Factor as P*A = L*U, leaving <paramref name="a"/> untouched. The result
    /// owns a copy of the factors.
    /// </summary>
    /// <param name="a">The matrix to factor. Not modified.</param>
    /// <param name="blockSize">Panel width; zero selects the default. The optimum shifts with size.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    public static LuDecomposition<double> FactorLu(
        this Matrix<double> a, int blockSize = 0, Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        // The copy is what keeps the caller's matrix intact; the in-place path
        // is Workspace.FactorLu, for a caller who has measured the copy.
        return (workspace ?? Workspace.Shared).FactorLu(a.Clone(), blockSize);
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
    ///
    /// An extension on the real decomposition only, for now: the complex one
    /// needs the complex norm estimator, and until that exists asking for it
    /// is a compile error rather than a run-time one.
    /// </summary>
    /// <param name="lu">The factorization.</param>
    /// <param name="columns">Probe columns for the estimator; more costs more products.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square.</exception>
    public static double ReciprocalCondition(this LuDecomposition<double> lu, int columns = NormEstimate.DefaultColumns)
    {
        ArgumentNullException.ThrowIfNull(lu);

        if (!lu.IsSquare)
            throw new InvalidOperationException("Condition estimation requires a square factorization.");

        return Condition.ReciprocalOne(lu.OneNormOfA, lu, columns);
    }

    /// <summary>
    /// Solve A*X = B for a general square A, by LU with partial pivoting.
    ///
    /// Factoring and discarding, so solving against several right-hand sides in
    /// separate calls does the O(n^3) work each time. Call
    /// <see cref="FactorLu(Matrix{double}, int, Workspace?)"/> once and reuse it instead, or pass all the
    /// right-hand sides as columns of <paramref name="b"/>.
    /// </summary>
    /// <param name="a">Coefficient matrix, square. Not modified.</param>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <exception cref="ArgumentException">A is not square, or the shapes disagree.</exception>
    /// <exception cref="InvalidOperationException">A has an exactly zero pivot.</exception>
    public static Matrix<double> Solve(
        this Matrix<double> a, ReadOnlyMatrixView<double> b, Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"Solve requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        return a.FactorLu(workspace: workspace).Solve(b);
    }

    /// <summary>||A||_1, the largest absolute column sum. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double OneNorm(this ReadOnlyMatrixView<double> a) => KernelEntry.OneNorm(a.ToOperand());

    /// <summary>||A||_1, the largest absolute column sum. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double OneNorm(this Matrix<double> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        return KernelEntry.OneNorm(a.ReadOnlyView.ToOperand());
    }

    /// <summary>||A||_inf, the largest absolute row sum. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double InfinityNorm(this ReadOnlyMatrixView<double> a) => KernelEntry.InfinityNorm(a.ToOperand());

    /// <summary>||A||_inf, the largest absolute row sum. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double InfinityNorm(this Matrix<double> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        return KernelEntry.InfinityNorm(a.ReadOnlyView.ToOperand());
    }

    /// <summary>||A||_F, the square root of the sum of squares. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double FrobeniusNorm(this ReadOnlyMatrixView<double> a) => KernelEntry.FrobeniusNorm(a.ToOperand());

    /// <summary>||A||_F, the square root of the sum of squares. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double FrobeniusNorm(this Matrix<double> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        return KernelEntry.FrobeniusNorm(a.ReadOnlyView.ToOperand());
    }

    /// <summary>
    /// Estimate ||A^power||_1 without forming the power, by Higham and
    /// Tisseur's block algorithm.
    ///
    /// For <c>power</c> of 1 this is slower than <see cref="OneNorm(Matrix{double})"/>
    /// and no more accurate — the exact norm of a dense matrix is already
    /// O(n^2). It earns its place when the power is greater than one, where
    /// forming A^k would cost k products, and that is the quantity
    /// scaling-and-squaring needs.
    ///
    /// The result is a lower bound. It is exact for most matrices and rarely
    /// off by more than a factor of two, but nothing at run time distinguishes
    /// an exact answer from an underestimate.
    /// </summary>
    /// <param name="a">The matrix to measure. Must be square.</param>
    /// <param name="power">How many times to apply A. At least 1.</param>
    /// <param name="columns">Probe columns; more costs more products and estimates better.</param>
    /// <exception cref="ArgumentException">The matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The power is less than 1.</exception>
    public static double EstimateOneNorm(this Matrix<double> a, int power = 1, int columns = NormEstimate.DefaultColumns)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"Norm estimation requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        return NormEstimate.Of(new DenseMatrixOperator(a, power), columns).Value;
    }
}

/// <summary>
/// Solves against a matrix whose shape is known at compile time.
///
/// These are the payoff for carrying a structure in the type: the substitution
/// routine is chosen by the type parameter, so no factorization is performed,
/// no run-time branch is taken, and the wrong triangle cannot be read.
/// </summary>
public static class StructuredSolveExtensions
{
    /// <summary>Solve A*X = B by substitution, returning a fresh X.</summary>
    /// <typeparam name="TStructure">The triangular structure of A.</typeparam>
    /// <param name="a">The triangular operand, order n.</param>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <exception cref="ArgumentException">A is not square, or the shapes disagree.</exception>
    public static Matrix<double> Solve<TStructure>(
        this StructuredMatrix<double, TStructure> a, ReadOnlyMatrixView<double> b)
        where TStructure : ITriangularStructure
    {
        var x = Matrix.From(b);
        a.SolveInPlace(x.View);
        return x;
    }

    /// <summary>Solve A*X = B by substitution, overwriting <paramref name="b"/> with X.</summary>
    /// <typeparam name="TStructure">The triangular structure of A.</typeparam>
    /// <param name="a">The triangular operand, order n.</param>
    /// <param name="b">Right-hand sides on entry, the solution on exit.</param>
    /// <exception cref="ArgumentException">A is not square, or the shapes disagree.</exception>
    public static void SolveInPlace<TStructure>(
        this StructuredMatrix<double, TStructure> a, MatrixView<double> b)
        where TStructure : ITriangularStructure
    {
        Validate<TStructure>(a, b);
        TStructure.SolveInPlace(a.View, b);
    }

    /// <summary>Solve A^T*X = B by substitution, returning a fresh X.</summary>
    /// <typeparam name="TStructure">The triangular structure of A.</typeparam>
    /// <param name="a">The triangular operand, order n.</param>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <exception cref="ArgumentException">A is not square, or the shapes disagree.</exception>
    public static Matrix<double> SolveTransposed<TStructure>(
        this StructuredMatrix<double, TStructure> a, ReadOnlyMatrixView<double> b)
        where TStructure : ITriangularStructure
    {
        var x = Matrix.From(b);
        a.SolveTransposedInPlace(x.View);
        return x;
    }

    /// <summary>Solve A^T*X = B by substitution, overwriting <paramref name="b"/> with X.</summary>
    /// <typeparam name="TStructure">The triangular structure of A.</typeparam>
    /// <param name="a">The triangular operand, order n.</param>
    /// <param name="b">Right-hand sides on entry, the solution on exit.</param>
    /// <exception cref="ArgumentException">A is not square, or the shapes disagree.</exception>
    public static void SolveTransposedInPlace<TStructure>(
        this StructuredMatrix<double, TStructure> a, MatrixView<double> b)
        where TStructure : ITriangularStructure
    {
        Validate<TStructure>(a, b);
        TStructure.SolveTransposedInPlace(a.View, b);
    }

    private static void Validate<TStructure>(StructuredMatrix<double, TStructure> a, MatrixView<double> b)
        where TStructure : ITriangularStructure
    {
        if (a.Rows != a.Columns)
        {
            throw new ArgumentException(
                $"A {TStructure.Name} solve requires a square operand, got {a.Rows}x{a.Columns}.", nameof(a));
        }

        if (b.Rows != a.Rows)
            throw new ArgumentException($"Right-hand side has {b.Rows} rows, expected {a.Rows}.", nameof(b));
    }
}
