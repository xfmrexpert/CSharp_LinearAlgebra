using System.Numerics;

namespace Tensile;

/// <summary>
/// <see cref="IElementKernels{T}"/> for <see cref="Complex"/>.
///
/// Multiplication is the 4M method: split each operand into its real and
/// imaginary parts and form the product from four real GEMMs,
///
///     Re(AB) = Ar*Br - Ai*Bi,    Im(AB) = Ar*Bi + Ai*Br.
///
/// Four real GEMMs of order n are 8n^3 real flops, exactly the count of a
/// conventional complex product, so nothing is wasted -- and every one of them
/// runs on the real GEMM that is already verified against BLIS and already
/// guarded by the codegen gate. No new micro-kernel, no new unsafe code.
///
/// Its error bound is the conventional product's too, entry by entry: each
/// part is a signed sum of exactly the products the conventional method forms,
/// only associated differently. That is the property the 3M method gives up.
/// 3M saves a quarter of the flops by computing Im(AB) as
/// (Ar + Ai)(Br + Bi) - Ar*Br - Ai*Bi, whose rounding error is relative to the
/// whole magnitude of the operands rather than to the imaginary parts, so a
/// small imaginary component can be lost entirely. In a lossy transmission
/// line the small component is often the damping, which is precisely what a
/// transient model exists to get right; the test suite has a case built to
/// fail under 3M.
///
/// The cost of splitting is the buffers: real and imaginary copies of A, B
/// and the product, O(mk + kn + mn) doubles against O(mnk) of work. The
/// workspace keeps them between products (<see cref="ComplexScratch"/>), up to
/// a cap, so a run of products -- the trailing updates of a factorization, the
/// squarings of an exponential -- allocates once rather than every call. What
/// remains is the copying: one pass to split, one to combine, and each part
/// packed twice, once for each product it takes part in. Splitting inside the
/// packing routines instead -- BLIS's 1M method -- would remove the passes and
/// halve the B packing, and is deliberately left until profiling says the
/// remainder matters.
/// </summary>
internal readonly struct ComplexKernels : IElementKernels<Complex>
{
    /// <inheritdoc/>
    /// <exception cref="AllocationLimitException">
    /// A split buffer exceeds <see cref="TensileLimits.MaxElements"/>. Each has
    /// the element count of an operand or of C, so this happens only if C is a
    /// caller's view larger than the library would allocate.
    /// </exception>
    /// <exception cref="ObjectDisposedException">The workspace has been disposed.</exception>
    public static void Multiply(
        Workspace workspace,
        ReadOnlyMatrixView<Complex> a,
        ReadOnlyMatrixView<Complex> b,
        MatrixView<Complex> c,
        Complex alpha,
        Complex beta)
    {
        ArgumentNullException.ThrowIfNull(workspace);

        if (a.Columns != b.Rows)
        {
            throw new ArgumentException(
                $"Inner dimensions disagree: A is {a.Rows}x{a.Columns}, B is {b.Rows}x{b.Columns}.", nameof(b));
        }

        if (c.Rows != a.Rows || c.Columns != b.Columns)
        {
            throw new ArgumentException(
                $"Destination is {c.Rows}x{c.Columns}, expected {a.Rows}x{b.Columns}.", nameof(c));
        }

        // A destination with no elements has nothing to receive, however many
        // columns it nominally has (finding 11). An empty inner dimension is
        // different -- C is real and must still be scaled by beta -- so it goes
        // through, and the four products come out as zeros.
        if (c.Rows == 0 || c.Columns == 0) return;

        int m = a.Rows, k = a.Columns, n = b.Columns;

        // Held across all four products and the combine: the buffers belong
        // to the workspace, and another caller's product must not land in
        // them between two of ours. Taken once; the products go through
        // MultiplyHeld, which does not take it again.
        lock (workspace.Gate)
        {
            SplitBuffers buffers = workspace.ComplexScratch.Acquire(m, k, n);

            MatrixView<double> ar = buffers.RealA(m, k), ai = buffers.ImaginaryA(m, k);
            MatrixView<double> br = buffers.RealB(k, n), bi = buffers.ImaginaryB(k, n);
            MatrixView<double> real = buffers.RealProduct(m, n), imaginary = buffers.ImaginaryProduct(m, n);

            // Every buffer may hold a previous product's values. Split writes
            // every element of the four operand parts, and the first product
            // into each part of the result has beta = 0, which overwrites --
            // so nothing stale is ever read, and nothing needs zeroing.
            Split(a, ar, ai);
            Split(b, br, bi);

            workspace.MultiplyHeld(ar, br, real, alpha: 1.0, beta: 0.0);
            workspace.MultiplyHeld(ai, bi, real, alpha: -1.0, beta: 1.0);
            workspace.MultiplyHeld(ar, bi, imaginary, alpha: 1.0, beta: 0.0);
            workspace.MultiplyHeld(ai, br, imaginary, alpha: 1.0, beta: 1.0);

            // C is written only here, after all four products have succeeded, so a
            // failure in any of them -- a disposed workspace, a refused buffer --
            // leaves the caller's destination exactly as it was.
            Combine(real, imaginary, c, alpha, beta);
        }
    }

    /// <inheritdoc/>
    public static double Magnitude(Complex value) => Complex.Abs(value);

    /// <inheritdoc/>
    public static double PivotMagnitude(Complex value) => Math.Abs(value.Real) + Math.Abs(value.Imaginary);

    /// <inheritdoc/>
    public static Complex Conjugate(Complex value) => Complex.Conjugate(value);

    /// <inheritdoc/>
    public static Complex Sign(Complex value)
    {
        if (value == Complex.Zero) return Complex.One;

        double modulus = Complex.Abs(value);
        return new Complex(value.Real / modulus, value.Imaginary / modulus);
    }

    /// <inheritdoc/>
    public static bool SignsAreDiscrete => false;

    /// <inheritdoc/>
    public static Complex Scale(Complex value, double factor) => new(value.Real * factor, value.Imaginary * factor);

    /// <inheritdoc/>
    public static Complex Exp(Complex value) => Complex.Exp(value);

    /// <inheritdoc/>
    public static IAdjointOperator<Complex> PowerOperator(Matrix<Complex> a, int power) => new ComplexDenseOperator(a, power);

    /// <inheritdoc/>
    public static LuDecomposition<Complex> FactorLuInPlace(Workspace workspace, Matrix<Complex> a) => workspace.FactorLu(a);

    /// <inheritdoc/>
    public static double OneNorm(ReadOnlyMatrixView<Complex> a)
    {
        if (a.Rows == 0 || a.Columns == 0) return 0.0;

        double best = 0.0;

        for (int j = 0; j < a.Columns; j++)
        {
            ReadOnlySpan<Complex> column = a.Column(j);

            double sum = 0.0;
            for (int i = 0; i < column.Length; i++) sum += Complex.Abs(column[i]);

            best = Math.Max(best, sum);
        }

        return best;
    }

    /// <inheritdoc/>
    public static double InfinityNorm(ReadOnlyMatrixView<Complex> a)
    {
        if (a.Rows == 0 || a.Columns == 0) return 0.0;

        double[] sums = Storage.Array<double>(a.Rows, "a row-sum work vector");

        for (int j = 0; j < a.Columns; j++)
        {
            ReadOnlySpan<Complex> column = a.Column(j);
            for (int i = 0; i < column.Length; i++) sums[i] += Complex.Abs(column[i]);
        }

        double best = 0.0;
        for (int i = 0; i < sums.Length; i++) best = Math.Max(best, sums[i]);

        return best;
    }

    /// <summary>The real and imaginary parts of a complex view, written into two real views of its shape.</summary>
    private static void Split(ReadOnlyMatrixView<Complex> source, MatrixView<double> real, MatrixView<double> imaginary)
    {
        for (int j = 0; j < source.Columns; j++)
        {
            ReadOnlySpan<Complex> column = source.Column(j);
            Span<double> re = real.Column(j);
            Span<double> im = imaginary.Column(j);

            for (int i = 0; i < column.Length; i++)
            {
                re[i] = column[i].Real;
                im[i] = column[i].Imaginary;
            }
        }
    }

    /// <summary>
    /// C := beta*C + alpha*(P + iQ), with the common scalars special-cased.
    ///
    /// Not only for speed. A general complex product with alpha = 1 computes
    /// 1*p - 0*q for the real part, and if q is infinite, 0*q is NaN, which
    /// would put a NaN into a real part that should have been p. Taking
    /// alpha = 1 and a real alpha exactly avoids manufacturing that. And beta = 0
    /// overwrites rather than scales, the contract every multiply here keeps,
    /// so a destination full of NaN still yields a finite result.
    /// </summary>
    private static void Combine(
        ReadOnlyMatrixView<double> real, ReadOnlyMatrixView<double> imaginary, MatrixView<Complex> c, Complex alpha, Complex beta)
    {
        bool unitAlpha = alpha == Complex.One;
        bool realAlpha = alpha.Imaginary == 0.0;
        bool zeroBeta = beta == Complex.Zero;
        bool unitBeta = beta == Complex.One;

        for (int j = 0; j < c.Columns; j++)
        {
            ReadOnlySpan<double> p = real.Column(j);
            ReadOnlySpan<double> q = imaginary.Column(j);
            Span<Complex> target = c.Column(j);

            for (int i = 0; i < target.Length; i++)
            {
                Complex product =
                    unitAlpha ? new Complex(p[i], q[i])
                    : realAlpha ? new Complex(alpha.Real * p[i], alpha.Real * q[i])
                    : alpha * new Complex(p[i], q[i]);

                target[i] =
                    zeroBeta ? product
                    : unitBeta ? target[i] + product
                    : (beta * target[i]) + product;
            }
        }
    }
}
