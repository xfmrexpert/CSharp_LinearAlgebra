using System.Numerics;
using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// Something that can be applied to a panel of columns without being asked for
/// its entries. That is the point of the abstraction -- the uses are A^k (where
/// forming the power would cost k GEMMs), A^-1 (where forming it would cost an
/// explicit inverse, something no well-behaved library computes), and a
/// matrix-free operator that has no entries at all. This is the extension point
/// <c>expm</c> and <c>expmv</c> plug into, and the one a FEM or MTL operator
/// plugs into without ever assembling a dense matrix.
///
/// Generic over the element type because the same operator shape serves real
/// and complex problems. An element type the library has no arithmetic for is
/// not an error here -- nothing in this interface computes anything -- but no
/// algorithm will accept it.
///
/// Only the forward direction is required here. Applying the adjoint is a
/// genuinely separate capability -- a matrix-free operator can very often
/// apply A and not cheaply apply A^H -- so it lives on
/// <see cref="IAdjointOperator{T}"/>, which is what the 1-norm estimator asks
/// for. Requiring both on one interface would force every implementer to
/// supply an adjoint so that one algorithm could have it.
///
/// Both panels arrive as bound views, so an implementation receives their
/// extents along with their contents and cannot be handed a buffer that is
/// shorter than <see cref="Order"/> claims. An implementation should still
/// check that <c>x.Rows</c> and <c>y.Rows</c> equal its order, and that
/// <c>x.Columns == y.Columns</c>, and reject anything else as an argument;
/// the estimator always satisfies both, and any other caller may not.
///
/// Square operators only.
/// </summary>
/// <typeparam name="T">The element type: <see cref="double"/> or <see cref="Complex"/> for anything the library computes with.</typeparam>
public interface ILinearOperator<T> where T : unmanaged, INumberBase<T>
{
    /// <summary>Order n of the operator.</summary>
    int Order { get; }

    /// <summary>Y := A * X, X and Y being n x t panels of the same width.</summary>
    /// <param name="x">The panel to apply the operator to.</param>
    /// <param name="y">Receives the result. Overwritten.</param>
    void Apply(ReadOnlyMatrixView<T> x, MatrixView<T> y);
}

/// <summary>
/// An operator that can also apply its adjoint, A^H -- the conjugate
/// transpose.
///
/// Higham and Tisseur's 1-norm estimator alternates products with A and with
/// its adjoint -- it is the adjoint step that tells it which unit vectors to
/// try next -- so <see cref="NormEstimate"/> requires this rather than the bare
/// <see cref="ILinearOperator{T}"/>. Anything that only ever applies A
/// forward, which includes <c>expmv</c>, should take the weaker interface.
///
/// Adjoint and not transpose, because that is what the estimator needs. For a
/// real operator the two are the same operation, which is why this was once
/// named for the transpose; for a complex one the transpose is the wrong
/// operation entirely, and an implementation that supplied A^T where A^H was
/// meant would give an estimator that is quietly wrong rather than one that
/// fails.
///
/// Both directions must map the same space, so the operator is square in the
/// sense <see cref="ILinearOperator{T}.Order"/> already requires.
/// </summary>
/// <typeparam name="T">The element type.</typeparam>
public interface IAdjointOperator<T> : ILinearOperator<T> where T : unmanaged, INumberBase<T>
{
    /// <summary>Y := A^H * X, X and Y being n x t panels of the same width. For a real operator, A^T.</summary>
    /// <param name="x">The panel to apply the adjoint to.</param>
    /// <param name="y">Receives the result. Overwritten.</param>
    void ApplyAdjoint(ReadOnlyMatrixView<T> x, MatrixView<T> y);
}

/// <summary>
/// A dense matrix, optionally raised to a power: applying this operator
/// computes A^p * X by p successive panel products, never forming A^p.
///
/// The power is what Al-Mohy and Higham's scaling-and-squaring needs. It
/// chooses the scaling from estimates of ||A^k||^(1/k) rather than from ||A||,
/// and the point of doing so is lost if you form A^k to measure it.
///
/// The operator holds a reference to the matrix, not a copy, so later writes
/// to the matrix are seen by later applications. It holds no other state and
/// is safe to apply from several threads at once.
/// </summary>
public sealed class DenseMatrixOperator : IAdjointOperator<double>
{
    private readonly Matrix<double> _a;
    private readonly int _power;

    /// <summary>Wrap a square matrix, to be applied <paramref name="power"/> times.</summary>
    /// <param name="a">The matrix. Referenced, not copied.</param>
    /// <param name="power">How many times to apply A. At least 1.</param>
    /// <exception cref="ArgumentException">The matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The power is less than 1.</exception>
    public DenseMatrixOperator(Matrix<double> a, int power = 1)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentOutOfRangeException.ThrowIfLessThan(power, 1);

        if (!a.IsSquare)
            throw new ArgumentException($"A linear operator must be square, got {a.Rows}x{a.Columns}.", nameof(a));

        _a = a;
        _power = power;
    }

    /// <inheritdoc/>
    public int Order => _a.Rows;

    /// <summary>How many times the matrix is applied.</summary>
    public int Power => _power;

    /// <inheritdoc/>
    public void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y) => Repeat(x, y, transposed: false);

    /// <inheritdoc/>
    public void ApplyAdjoint(ReadOnlyMatrixView<double> x, MatrixView<double> y) => Repeat(x, y, transposed: true);

    /// <summary>
    /// Apply A (or A^T) <see cref="Power"/> times, landing in Y.
    ///
    /// (A^p)^T = (A^T)^p, so the transposed case is the same loop with the
    /// transposed product. Each subsequent product copies Y aside first: that
    /// is O(n*t) against the O(n^2*t) of the product itself, and it keeps the
    /// result in the caller's panel without depending on the parity of p. The
    /// staging panel is allocated per call rather than kept, which is what
    /// makes the operator stateless and therefore shareable.
    /// </summary>
    private void Repeat(ReadOnlyMatrixView<double> x, MatrixView<double> y, bool transposed)
    {
        Validate(x, y);
        Product(x, y, transposed);

        if (_power == 1) return;

        var staging = new Matrix<double>(Order, x.Columns);

        for (int step = 1; step < _power; step++)
        {
            y.CopyTo(staging.View);
            Product(staging.ReadOnlyView, y, transposed);
        }
    }

    private void Product(ReadOnlyMatrixView<double> x, MatrixView<double> y, bool transposed)
    {
        if (transposed) KernelEntry.MultiplyPanelTransposed(_a.ReadOnlyView.ToOperand(), x.ToOperand(), y.ToTarget());
        else KernelEntry.MultiplyPanel(_a.ReadOnlyView.ToOperand(), x.ToOperand(), y.ToTarget());
    }

    private void Validate(ReadOnlyMatrixView<double> x, MatrixView<double> y)
    {
        if (x.Rows != Order)
            throw new ArgumentException($"Panel has {x.Rows} rows, expected the operator's order {Order}.", nameof(x));

        if (y.Rows != Order || y.Columns != x.Columns)
        {
            throw new ArgumentException(
                $"Result panel is {y.Rows}x{y.Columns}, expected {Order}x{x.Columns}.", nameof(y));
        }
    }
}

/// <summary>
/// A dense complex matrix raised to a power, as an operator: the complex
/// counterpart of <see cref="DenseMatrixOperator"/>, for the complex norm
/// estimate. Internal, because the public shape of a complex dense operator is
/// a decision that the native complex exponential should make rather than
/// this one.
///
/// Each product is four real panel products on the split parts, the 4M
/// method at panel width:
///
///     A X   = (Ar Xr - Ai Xi) + i (Ar Xi + Ai Xr)
///     A^H X = (Ar^T Xr + Ai^T Xi) + i (Ar^T Xi - Ai^T Xr)
///
/// so the streamed real kernels do the O(n^2 t) work and the conjugate
/// transpose needs no kernel of its own. Unlike the real operator this one
/// takes a snapshot: A is split once, at construction, because splitting it
/// again for every product would cost O(n^2) against O(n^2 t) of work. Later
/// writes to the matrix are therefore not seen.
/// </summary>
internal sealed class ComplexDenseOperator : IAdjointOperator<Complex>
{
    private readonly Matrix<double> _real;
    private readonly Matrix<double> _imaginary;
    private readonly int _power;

    /// <exception cref="ArgumentException">The matrix is not square.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The power is less than 1.</exception>
    public ComplexDenseOperator(Matrix<Complex> a, int power = 1)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentOutOfRangeException.ThrowIfLessThan(power, 1);

        if (!a.IsSquare)
            throw new ArgumentException($"A linear operator must be square, got {a.Rows}x{a.Columns}.", nameof(a));

        _real = new Matrix<double>(a.Rows, a.Columns);
        _imaginary = new Matrix<double>(a.Rows, a.Columns);
        Split(a.ReadOnlyView, _real.View, _imaginary.View);
        _power = power;
    }

    /// <inheritdoc/>
    public int Order => _real.Rows;

    /// <inheritdoc/>
    public void Apply(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y) => Repeat(x, y, adjoint: false);

    /// <inheritdoc/>
    public void ApplyAdjoint(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y) => Repeat(x, y, adjoint: true);

    /// <summary>(A^p)^H = (A^H)^p, so both directions are the same loop over one product.</summary>
    private void Repeat(ReadOnlyMatrixView<Complex> x, MatrixView<Complex> y, bool adjoint)
    {
        if (x.Rows != Order)
            throw new ArgumentException($"Panel has {x.Rows} rows, expected the operator's order {Order}.", nameof(x));

        if (y.Rows != Order || y.Columns != x.Columns)
            throw new ArgumentException($"Result panel is {y.Rows}x{y.Columns}, expected {Order}x{x.Columns}.", nameof(y));

        if (Order == 0 || x.Columns == 0) return;

        // Six n x t real panels per call, on the ordinary heap: the
        // estimator applies an operator a handful of times, and the pinned
        // heap, which is never compacted, is for storage that lives.
        var shape = new MatrixShape(Order, x.Columns);
        string purpose = $"an {Order}x{x.Columns} split panel";
        MatrixView<double> xr = Panel(shape, purpose), xi = Panel(shape, purpose);
        MatrixView<double> p = Panel(shape, purpose), q = Panel(shape, purpose);
        MatrixView<double> r = Panel(shape, purpose), u = Panel(shape, purpose);

        Split(x, xr, xi);

        for (int step = 0; step < _power; step++)
        {
            if (step > 0) Split(y, xr, xi);

            Product(_real.ReadOnlyView, xr, p, adjoint);        // Ar Xr
            Product(_imaginary.ReadOnlyView, xi, q, adjoint);   // Ai Xi
            Product(_real.ReadOnlyView, xi, r, adjoint);        // Ar Xi
            Product(_imaginary.ReadOnlyView, xr, u, adjoint);   // Ai Xr

            for (int j = 0; j < x.Columns; j++)
            {
                ReadOnlySpan<double> pj = p.Column(j), qj = q.Column(j);
                ReadOnlySpan<double> rj = r.Column(j), uj = u.Column(j);
                Span<Complex> target = y.Column(j);

                for (int i = 0; i < target.Length; i++)
                {
                    target[i] = adjoint
                        ? new Complex(pj[i] + qj[i], rj[i] - uj[i])
                        : new Complex(pj[i] - qj[i], rj[i] + uj[i]);
                }
            }
        }
    }

    private static MatrixView<double> Panel(MatrixShape shape, string purpose) =>
        MatrixView<double>.Bind(Storage.Array<double>(shape.RequiredExtent, purpose), shape);

    private static void Product(ReadOnlyMatrixView<double> a, ReadOnlyMatrixView<double> x, MatrixView<double> y, bool transposed)
    {
        if (transposed) KernelEntry.MultiplyPanelTransposed(a.ToOperand(), x.ToOperand(), y.ToTarget());
        else KernelEntry.MultiplyPanel(a.ToOperand(), x.ToOperand(), y.ToTarget());
    }

    private static void Split(ReadOnlyMatrixView<Complex> source, MatrixView<double> real, MatrixView<double> imaginary)
    {
        for (int j = 0; j < source.Columns; j++)
        {
            ReadOnlySpan<Complex> column = source.Column(j);
            Span<double> re = real.Column(j), im = imaginary.Column(j);

            for (int i = 0; i < column.Length; i++)
            {
                re[i] = column[i].Real;
                im[i] = column[i].Imaginary;
            }
        }
    }
}

/// <summary>
/// The inverse of an LU-factored matrix, as an operator: applying it solves
/// rather than multiplying. This is what turns the 1-norm estimator into a
/// condition estimator, since cond_1(A) = ||A||_1 * ||A^-1||_1 and the second
/// factor is exactly what the estimator can reach without forming A^-1. Its
/// adjoint is (A^-1)^H = (A^H)^-1, which is the adjoint solve.
/// </summary>
/// <typeparam name="T">The element type of the factorization.</typeparam>
public sealed class LuInverseOperator<T> : IAdjointOperator<T> where T : unmanaged, INumberBase<T>
{
    private readonly LuDecomposition<T> _lu;

    /// <summary>Wrap a factorization so the estimator can probe A^-1.</summary>
    /// <param name="lu">A square factorization.</param>
    /// <exception cref="ArgumentException">The factorization is not square.</exception>
    public LuInverseOperator(LuDecomposition<T> lu)
    {
        ArgumentNullException.ThrowIfNull(lu);

        if (!lu.IsSquare)
            throw new ArgumentException("An inverse operator requires a square factorization.", nameof(lu));

        _lu = lu;
    }

    /// <inheritdoc/>
    public int Order => _lu.Rows;

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">The factorization has an exactly zero pivot.</exception>
    public void Apply(ReadOnlyMatrixView<T> x, MatrixView<T> y)
    {
        CopyInto(x, y);
        _lu.SolveInPlace(y);
    }

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">The factorization has an exactly zero pivot.</exception>
    public void ApplyAdjoint(ReadOnlyMatrixView<T> x, MatrixView<T> y)
    {
        CopyInto(x, y);
        _lu.SolveAdjointInPlace(y);
    }

    /// <summary>The solves work in place, so the right-hand side has to arrive in Y.</summary>
    private static void CopyInto(ReadOnlyMatrixView<T> x, MatrixView<T> y)
    {
        if (y.Rows != x.Rows || y.Columns != x.Columns)
        {
            throw new ArgumentException(
                $"Result panel is {y.Rows}x{y.Columns}, expected {x.Rows}x{x.Columns}.", nameof(y));
        }

        for (int j = 0; j < x.Columns; j++) x.Column(j).CopyTo(y.Column(j));
    }
}
