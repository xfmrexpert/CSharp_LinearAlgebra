using System.Numerics;

namespace Tensile;

// The complex exponential. A partial of MatrixExponential so that the
// implementation could change behind a signature that did not move: it was
// first computed through the real embedding, and is now the same algorithm as
// the real one, run natively on complex arithmetic. The embedding survives as
// its oracle, in ComplexEmbedding.
public static partial class MatrixExponential
{
    /// <summary>
    /// exp(A) for a complex matrix -- the matrix exponential, not the
    /// element-wise one.
    ///
    /// The same scaling-and-squaring algorithm as the real overload, run in
    /// complex arithmetic: products are 4M complex products on the workspace's
    /// real GEMM, the Padé denominator is factored by the complex LU, and the
    /// ||A^k||^(1/k) estimates come from the complex norm estimator. The
    /// <c>ell</c> test uses the matrix of moduli, as the paper's complex case
    /// does. Against computing through the 2n x 2n real representation, which
    /// is how this was first done, that halves the flops and the memory, and
    /// the scaling is chosen from the complex norms rather than from real
    /// ones up to sqrt(2) larger.
    /// </summary>
    /// <param name="a">The matrix to exponentiate. Must be square. Not modified.</param>
    /// <param name="workspace">Buffers and kernel choice; null uses <see cref="Workspace.Shared"/>.</param>
    /// <returns>A new matrix holding exp(A).</returns>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    /// <exception cref="ArgumentException">The matrix is not square.</exception>
    /// <exception cref="InvalidOperationException">
    /// The Padé denominator was exactly singular, which indicates an input
    /// that is not finite.
    /// </exception>
    public static Matrix<Complex> Expm(this Matrix<Complex> a, Workspace? workspace = null) =>
        Expm(a, workspace, diagnostics: null);

    /// <summary>The complex implementation, with an optional report of the branch taken.</summary>
    internal static Matrix<Complex> Expm(Matrix<Complex> a, Workspace? workspace, ExpmDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"The matrix exponential requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        if (a.Rows <= 1)
        {
            Report(diagnostics, degree: 0, squarings: 0);
            return Scalar<Complex, ComplexKernels>(a);
        }

        return ScalingAndSquaring<Complex, ComplexKernels>(a, workspace ?? Workspace.Shared, diagnostics);
    }
}
