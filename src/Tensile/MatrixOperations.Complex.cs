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
}
