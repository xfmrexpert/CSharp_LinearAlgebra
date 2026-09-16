using System.Runtime.InteropServices;

namespace GemmLab;

/// <summary>
/// Something the 1-norm estimator can probe. Both directions are required: the
/// algorithm alternates products with A and with A^T, and it is the A^T step
/// that tells it which unit vectors to try next.
///
/// The operator is never asked for its entries, which is the point — the two
/// uses are A^k (where forming the power would cost k GEMMs) and A^-1 (where
/// forming it would cost an explicit inverse, something no well-behaved library
/// computes).
///
/// Square operators only; the estimator needs A and A^T to map the same space.
/// </summary>
public unsafe interface ILinearOperator
{
    /// <summary>Order n of the operator.</summary>
    int Order { get; }

    /// <summary>Y := A * X, X and Y being n x t column-major.</summary>
    void Apply(int t, double* x, int ldx, double* y, int ldy);

    /// <summary>Y := A^T * X, X and Y being n x t column-major.</summary>
    void ApplyTranspose(int t, double* x, int ldx, double* y, int ldy);
}

/// <summary>
/// A dense matrix, optionally raised to a power: applying this operator
/// computes A^p * X by p successive panel products, never forming A^p.
///
/// The power is what Al-Mohy and Higham's scaling-and-squaring needs. It
/// chooses the scaling from estimates of ||A^k||^(1/k) rather than from ||A||,
/// and the point of doing so is lost if you form A^k to measure it.
/// </summary>
public sealed unsafe class DenseMatrixOperator : ILinearOperator, IDisposable
{
    private readonly double* _a;
    private readonly int _n;
    private readonly int _lda;
    private readonly int _power;

    private double* _work;
    private int _workColumns;

    /// <param name="power">How many times to apply A. Must be at least 1.</param>
    public DenseMatrixOperator(int n, double* a, int lda, int power = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(n);
        ArgumentOutOfRangeException.ThrowIfLessThan(power, 1);

        _n = n;
        _a = a;
        _lda = lda;
        _power = power;
    }

    public int Order => _n;

    public void Apply(int t, double* x, int ldx, double* y, int ldy) =>
        Repeat(t, x, ldx, y, ldy, transposed: false);

    public void ApplyTranspose(int t, double* x, int ldx, double* y, int ldy) =>
        Repeat(t, x, ldx, y, ldy, transposed: true);

    /// <summary>
    /// Apply A (or A^T) <see cref="_power"/> times, landing in Y.
    ///
    /// (A^p)^T = (A^T)^p, so the transposed case is the same loop with the
    /// transposed product. Rather than ping-pong between two buffers, each
    /// subsequent product copies Y aside first: that is O(n*t) against the
    /// O(n^2*t) of the product itself, and it keeps the result in the caller's
    /// buffer without depending on the parity of p.
    /// </summary>
    private void Repeat(int t, double* x, int ldx, double* y, int ldy, bool transposed)
    {
        Product(t, x, ldx, y, ldy, transposed);

        if (_power == 1) return;

        EnsureWork(t);

        for (int step = 1; step < _power; step++)
        {
            for (int col = 0; col < t; col++)
            {
                new Span<double>(y + (nint)col * ldy, _n)
                    .CopyTo(new Span<double>(_work + (nint)col * _n, _n));
            }

            Product(t, _work, _n, y, ldy, transposed);
        }
    }

    private void Product(int t, double* x, int ldx, double* y, int ldy, bool transposed)
    {
        if (transposed) Blas2.MultiplyTransposed(_n, t, _a, _lda, x, ldx, y, ldy);
        else Blas2.Multiply(_n, t, _a, _lda, x, ldx, y, ldy);
    }

    private void EnsureWork(int t)
    {
        if (_work is not null && t <= _workColumns) return;

        if (_work is not null) NativeMemory.AlignedFree(_work);

        _work = (double*)NativeMemory.AlignedAlloc((nuint)_n * (nuint)t * sizeof(double), 64);
        _workColumns = t;
    }

    public void Dispose()
    {
        if (_work is not null) { NativeMemory.AlignedFree(_work); _work = null; }
        _workColumns = 0;
        GC.SuppressFinalize(this);
    }

    ~DenseMatrixOperator() => Dispose();
}

/// <summary>
/// The inverse of an LU-factored matrix, as an operator: applying it solves
/// rather than multiplying. This is what turns the 1-norm estimator into a
/// condition estimator, since cond_1(A) = ||A||_1 * ||A^-1||_1 and the second
/// factor is exactly what the estimator can reach without forming A^-1.
/// </summary>
public sealed unsafe class LuInverseOperator : ILinearOperator
{
    private readonly LuFactorization _lu;

    public LuInverseOperator(LuFactorization lu)
    {
        ArgumentNullException.ThrowIfNull(lu);

        if (!lu.IsSquare)
            throw new ArgumentException("Condition estimation requires a square factorization.", nameof(lu));

        _lu = lu;
    }

    public int Order => _lu.Rows;

    public void Apply(int t, double* x, int ldx, double* y, int ldy)
    {
        CopyInto(t, x, ldx, y, ldy);
        Lu.Solve(_lu, t, y, ldy);
    }

    public void ApplyTranspose(int t, double* x, int ldx, double* y, int ldy)
    {
        CopyInto(t, x, ldx, y, ldy);
        Lu.SolveTransposed(_lu, t, y, ldy);
    }

    /// <summary>The solves work in place, so the right-hand side arrives in Y.</summary>
    private void CopyInto(int t, double* x, int ldx, double* y, int ldy)
    {
        int n = Order;

        for (int col = 0; col < t; col++)
        {
            new Span<double>(x + (nint)col * ldx, n)
                .CopyTo(new Span<double>(y + (nint)col * ldy, n));
        }
    }
}
