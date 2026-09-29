using System.Numerics;

namespace Tensile;

// The fluent complex products, mirroring the real ones. Complex scalars have no
// compile-time constant form, so alpha and beta cannot be optional parameters
// here the way they are on the real overload; the plain product and the scaled
// accumulation are separate overloads instead.
public static partial class MatrixOperations
{
    /// <summary>A*B for complex operands, as a new matrix.</summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <exception cref="ArgumentException">The inner dimensions disagree.</exception>
    public static Matrix<Complex> Multiply(
        this Matrix<Complex> a, ReadOnlyMatrixView<Complex> b, Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        var result = new Matrix<Complex>(a.Rows, b.Columns);
        (workspace ?? Workspace.Shared).Multiply(a.ReadOnlyView, b, result.View);
        return result;
    }

    /// <summary>destination := A*B for complex operands, writing into storage the caller owns.</summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="destination">Destination, m x n. Overwritten.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <exception cref="ArgumentException">The shapes are not conformable.</exception>
    public static void MultiplyInto(
        this Matrix<Complex> a,
        ReadOnlyMatrixView<Complex> b,
        MatrixView<Complex> destination,
        Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        (workspace ?? Workspace.Shared).Multiply(a.ReadOnlyView, b, destination);
    }

    /// <summary>
    /// destination := beta*destination + alpha*A*B for complex operands,
    /// writing into storage the caller owns.
    /// </summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="destination">Destination, m x n.</param>
    /// <param name="alpha">Scalar on the product.</param>
    /// <param name="beta">Scalar on the existing contents of the destination. Zero overwrites rather than scales.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <exception cref="ArgumentException">The shapes are not conformable.</exception>
    public static void MultiplyInto(
        this Matrix<Complex> a,
        ReadOnlyMatrixView<Complex> b,
        MatrixView<Complex> destination,
        Complex alpha,
        Complex beta,
        Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        (workspace ?? Workspace.Shared).Multiply(a.ReadOnlyView, b, destination, alpha, beta);
    }

    /// <summary>
    /// Factor a complex matrix as P*A = L*U, leaving <paramref name="a"/>
    /// untouched. The result owns a copy of the factors. Pivoting maximises
    /// |Re| + |Im|, as <c>zgetrf</c> does.
    /// </summary>
    /// <param name="a">The matrix to factor. Not modified.</param>
    /// <param name="blockSize">Panel width; zero selects the default.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    public static LuDecomposition<Complex> FactorLu(
        this Matrix<Complex> a, int blockSize = 0, Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        return (workspace ?? Workspace.Shared).FactorLu(a.Clone(), blockSize);
    }

    /// <summary>
    /// Solve A*X = B for a general square complex A, by LU with partial
    /// pivoting. Factoring and discarding; to solve against several
    /// right-hand sides in separate calls, factor once and reuse it.
    /// </summary>
    /// <param name="a">Coefficient matrix, square. Not modified.</param>
    /// <param name="b">Right-hand sides, n x nrhs. Not modified.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <exception cref="ArgumentException">A is not square, or the shapes disagree.</exception>
    /// <exception cref="InvalidOperationException">A has an exactly zero pivot.</exception>
    public static Matrix<Complex> Solve(
        this Matrix<Complex> a, ReadOnlyMatrixView<Complex> b, Workspace? workspace = null)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"Solve requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        return a.FactorLu(workspace: workspace).Solve(b);
    }

    /// <summary>
    /// Estimate 1/cond_1(A) for a complex factorization, the equivalent of
    /// LAPACK's <c>zgecon</c>. Returns zero for an exactly singular
    /// factorization. Like the real estimate, an OVER-estimate of the
    /// reciprocal condition number: a small value reliably means
    /// ill-conditioning, a large one is weaker evidence of good conditioning.
    /// </summary>
    /// <param name="lu">The factorization.</param>
    /// <param name="columns">Probe columns for the estimator; more costs more products.</param>
    /// <exception cref="InvalidOperationException">The factorization is not square.</exception>
    public static double ReciprocalCondition(this LuDecomposition<Complex> lu, int columns = NormEstimate.DefaultColumns)
    {
        ArgumentNullException.ThrowIfNull(lu);

        if (!lu.IsSquare)
            throw new InvalidOperationException("Condition estimation requires a square factorization.");

        return Condition.ReciprocalOne(lu.OneNormOfA, lu, columns);
    }

    /// <summary>
    /// Estimate ||A^power||_1 for a complex matrix without forming the power,
    /// by the complex form of Higham and Tisseur's block algorithm. A lower
    /// bound, exact for most matrices and rarely off by more than a factor of
    /// two.
    /// </summary>
    /// <param name="a">The matrix to measure. Must be square.</param>
    /// <param name="power">How many times to apply A. At least 1.</param>
    /// <param name="columns">Probe columns; more costs more products and estimates better.</param>
    /// <exception cref="ArgumentException">The matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The power is less than 1.</exception>
    public static double EstimateOneNorm(this Matrix<Complex> a, int power = 1, int columns = NormEstimate.DefaultColumns)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"Norm estimation requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        return NormEstimate.Of(new ComplexDenseOperator(a, power), columns).Value;
    }

    /// <summary>||A||_1 for a complex matrix, the largest column sum of moduli. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double OneNorm(this Matrix<Complex> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        return ComplexKernels.OneNorm(a.ReadOnlyView);
    }

    /// <summary>||A||_inf for a complex matrix, the largest row sum of moduli. Exact, O(m*n).</summary>
    /// <param name="a">The matrix to measure.</param>
    public static double InfinityNorm(this Matrix<Complex> a)
    {
        ArgumentNullException.ThrowIfNull(a);
        return ComplexKernels.InfinityNorm(a.ReadOnlyView);
    }
}
