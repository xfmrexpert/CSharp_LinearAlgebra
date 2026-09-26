using System.Numerics;

namespace Tensile;

// The complex exponential. For now it is computed through the real embedding
// in ComplexEmbedding; when native complex primitives exist, this file is
// where the implementation changes and the public signature does not. That is
// why it is a partial of MatrixExponential rather than a class of its own.
public static partial class MatrixExponential
{
    /// <summary>
    /// exp(A) for a complex matrix -- the matrix exponential, not the
    /// element-wise one.
    ///
    /// Computed today by running the real algorithm on the 2n x 2n real
    /// matrix [[X, -Y], [Y, X]] that represents A = X + iY, which the
    /// exponential commutes with, and projecting the result back. The accuracy
    /// is the real algorithm's, verified independently here against a complex
    /// Taylor series.
    ///
    /// What that route costs against a native complex implementation: twice
    /// the flops, because every product of embedded matrices computes each
    /// block twice; twice the memory; and at most one extra squaring step,
    /// because the embedded 1-norm can exceed the complex one by up to
    /// sqrt(2). None of that affects the answer, only the time to reach it.
    /// </summary>
    /// <param name="a">The matrix to exponentiate. Must be square. Not modified.</param>
    /// <param name="workspace">Buffers and kernel choice for the real computation; null uses <see cref="Workspace.Shared"/>.</param>
    /// <returns>A new matrix holding exp(A).</returns>
    /// <exception cref="ArgumentNullException">The matrix is null.</exception>
    /// <exception cref="ArgumentException">The matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The 2n x 2n real representation is too large to own. It has four times
    /// the input's element count, so it can exceed a size the input did not.
    /// </exception>
    /// <exception cref="AllocationLimitException">
    /// The 2n x 2n real representation exceeds <see cref="TensileLimits.MaxElements"/>,
    /// which for the same reason can happen when the input itself fits.
    /// </exception>
    public static Matrix<Complex> Expm(this Matrix<Complex> a, Workspace? workspace = null) =>
        Expm(a, workspace, diagnostics: null);

    /// <summary>
    /// The complex implementation. The diagnostics report the real
    /// computation on the embedding, whose degree and squaring count are the
    /// ones the complex result actually used.
    /// </summary>
    internal static Matrix<Complex> Expm(Matrix<Complex> a, Workspace? workspace, ExpmDiagnostics? diagnostics)
    {
        ArgumentNullException.ThrowIfNull(a);

        if (!a.IsSquare)
            throw new ArgumentException($"The matrix exponential requires a square matrix, got {a.Rows}x{a.Columns}.", nameof(a));

        int n = a.Rows;

        if (n == 0)
        {
            Report(diagnostics, degree: 0, squarings: 0);
            return new Matrix<Complex>(0, 0);
        }

        // Scalar exp is exact; embedding it would run the algorithm on a 2x2.
        if (n == 1)
        {
            Report(diagnostics, degree: 0, squarings: 0);

            var scalar = new Matrix<Complex>(1, 1);
            scalar[0, 0] = Complex.Exp(a[0, 0]);
            return scalar;
        }

        var embedded = ComplexEmbedding.Embed(a);
        var result = Expm(embedded, workspace, diagnostics);

        return ComplexEmbedding.Project(result);
    }
}
