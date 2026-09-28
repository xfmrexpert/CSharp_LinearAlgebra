using System.Diagnostics;
using System.Numerics;

namespace Tensile;

// Complex products through a workspace. They run as four real products on this
// workspace's kernel and dispatch -- see ComplexKernels for the 4M method and
// why it, and not 3M -- so everything the workspace controls for real GEMM
// (which micro-kernel, serial or threaded) applies here unchanged.
public sealed partial class Workspace
{
    /// <summary>
    /// The split buffers complex products reuse, created on first use. Only
    /// under <see cref="Gate"/>: a disposed workspace throws here, before any
    /// buffer is handed out, so a failed product has touched nothing.
    /// </summary>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed.</exception>
    internal ComplexScratch ComplexScratch
    {
        get
        {
            Debug.Assert(_gate.IsHeldByCurrentThread, "Complex scratch requires the workspace lock.");

            _ = Active;
            return _complexScratch ??= new ComplexScratch();
        }
    }

    /// <summary>C := A*B for complex operands, with the shapes taken from the views.</summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="c">Destination, m x n. Overwritten.</param>
    /// <exception cref="ArgumentException">The shapes are not conformable.</exception>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed. The destination is left untouched.</exception>
    /// <exception cref="AllocationLimitException">A split buffer exceeds <see cref="TensileLimits.MaxElements"/>. The destination is left untouched.</exception>
    public void Multiply(ReadOnlyMatrixView<Complex> a, ReadOnlyMatrixView<Complex> b, MatrixView<Complex> c) =>
        ComplexKernels.Multiply(this, a, b, c, Complex.One, Complex.Zero);

    /// <summary>
    /// C := beta*C + alpha*A*B for complex operands.
    ///
    /// Computed as four real products (the 4M method), which does exactly the
    /// flops of a conventional complex product and has its error bound entry
    /// by entry. The destination is written only after all four have
    /// succeeded, so a failure leaves it as it was.
    /// </summary>
    /// <param name="a">Left operand, m x k.</param>
    /// <param name="b">Right operand, k x n.</param>
    /// <param name="c">Destination, m x n.</param>
    /// <param name="alpha">Scalar on the product.</param>
    /// <param name="beta">Scalar on the existing contents of C. Zero overwrites rather than scales, so a C full of NaN still yields a finite result.</param>
    /// <exception cref="ArgumentException">The shapes are not conformable.</exception>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed. The destination is left untouched.</exception>
    /// <exception cref="AllocationLimitException">A split buffer exceeds <see cref="TensileLimits.MaxElements"/>. The destination is left untouched.</exception>
    public void Multiply(
        ReadOnlyMatrixView<Complex> a,
        ReadOnlyMatrixView<Complex> b,
        MatrixView<Complex> c,
        Complex alpha,
        Complex beta) =>
        ComplexKernels.Multiply(this, a, b, c, alpha, beta);
}
