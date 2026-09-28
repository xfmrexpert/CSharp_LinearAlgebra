using System.Diagnostics;
using System.Numerics;
using Tensile.Kernels;

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

    /// <summary>
    /// Factor a complex <paramref name="a"/> in place as P*A = L*U, without
    /// copying; the complex counterpart of
    /// <see cref="FactorLu(Matrix{double}, int)"/>, with the same ownership
    /// rule: <paramref name="a"/> must not be written afterwards while the
    /// decomposition is in use.
    ///
    /// Pivoting maximises |Re| + |Im|, as LAPACK's <c>zgetrf</c> does, so a
    /// pivot sequence can be compared with one produced there. The trailing
    /// updates are 4M products on this workspace's kernel and dispatch; the
    /// panels and triangular solves are the same algorithm in safe code.
    /// </summary>
    /// <param name="a">The square or rectangular matrix to factor. Overwritten.</param>
    /// <param name="blockSize">Panel width; zero selects the default, which is the real factorization's measured default and has not been measured for complex.</param>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed. <paramref name="a"/> is untouched.</exception>
    public LuDecomposition<Complex> FactorLu(Matrix<Complex> a, int blockSize = 0) =>
        FactorLuManaged<Complex, ComplexKernels>(a, blockSize);

    /// <summary>
    /// The generic factorization for any element type with kernels. Complex
    /// uses it as its only path; the tests instantiate it for double too, as
    /// the oracle that the kernel factorization's pivots must match.
    /// </summary>
    internal LuDecomposition<T> FactorLuManaged<T, TKernels>(Matrix<T> a, int blockSize)
        where T : unmanaged, INumberBase<T>
        where TKernels : struct, IElementKernels<T>
    {
        ArgumentNullException.ThrowIfNull(a);

        // Before anything is written: the first place the factorization itself
        // would notice is its first trailing update, by which point the first
        // panel has already been overwritten.
        ThrowIfDisposed();

        // Captured before the factorization overwrites the matrix.
        double oneNorm = TKernels.OneNorm(a.ReadOnlyView);

        LuFactorization factorization = BlockedLu.Factor<T, TKernels>(this, a.View, blockSize);

        return new LuDecomposition<T>(a, factorization, oneNorm, ManagedLuSolver<T, TKernels>.Instance);
    }
}
