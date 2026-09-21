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
/// Only the forward direction is required here. Applying A^T is a genuinely
/// separate capability -- a matrix-free operator can very often apply A and not
/// cheaply apply A^T -- so it lives on <see cref="ITransposableOperator"/>,
/// which is what the 1-norm estimator asks for. Requiring both on one interface
/// would force every implementer to supply a transpose so that one algorithm
/// could have it.
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
public interface ILinearOperator
{
    /// <summary>Order n of the operator.</summary>
    int Order { get; }

    /// <summary>Y := A * X, X and Y being n x t panels of the same width.</summary>
    /// <param name="x">The panel to apply the operator to.</param>
    /// <param name="y">Receives the result. Overwritten.</param>
    void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y);
}

/// <summary>
/// An operator that can also apply its transpose.
///
/// Higham and Tisseur's 1-norm estimator alternates products with A and with
/// A^T -- it is the A^T step that tells it which unit vectors to try next -- so
/// <see cref="NormEstimate"/> requires this rather than the bare
/// <see cref="ILinearOperator"/>. Anything that only ever applies A forward,
/// which includes <c>expmv</c>, should take the weaker interface.
///
/// Both directions must map the same space, so the operator is square in the
/// sense <see cref="ILinearOperator.Order"/> already requires.
/// </summary>
public interface ITransposableOperator : ILinearOperator
{
    /// <summary>Y := A^T * X, X and Y being n x t panels of the same width.</summary>
    /// <param name="x">The panel to apply the transposed operator to.</param>
    /// <param name="y">Receives the result. Overwritten.</param>
    void ApplyTranspose(ReadOnlyMatrixView<double> x, MatrixView<double> y);
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
public sealed class DenseMatrixOperator : ITransposableOperator
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
    public void ApplyTranspose(ReadOnlyMatrixView<double> x, MatrixView<double> y) => Repeat(x, y, transposed: true);

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
/// The inverse of an LU-factored matrix, as an operator: applying it solves
/// rather than multiplying. This is what turns the 1-norm estimator into a
/// condition estimator, since cond_1(A) = ||A||_1 * ||A^-1||_1 and the second
/// factor is exactly what the estimator can reach without forming A^-1.
/// </summary>
public sealed class LuInverseOperator : ITransposableOperator
{
    private readonly LuDecomposition _lu;

    /// <summary>Wrap a factorization so the estimator can probe A^-1.</summary>
    /// <param name="lu">A square factorization.</param>
    /// <exception cref="ArgumentException">The factorization is not square.</exception>
    public LuInverseOperator(LuDecomposition lu)
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
    public void Apply(ReadOnlyMatrixView<double> x, MatrixView<double> y)
    {
        CopyInto(x, y);
        _lu.SolveInPlace(y);
    }

    /// <inheritdoc/>
    /// <exception cref="InvalidOperationException">The factorization has an exactly zero pivot.</exception>
    public void ApplyTranspose(ReadOnlyMatrixView<double> x, MatrixView<double> y)
    {
        CopyInto(x, y);
        _lu.SolveTransposedInPlace(y);
    }

    /// <summary>The solves work in place, so the right-hand side has to arrive in Y.</summary>
    private static void CopyInto(ReadOnlyMatrixView<double> x, MatrixView<double> y)
    {
        if (y.Rows != x.Rows || y.Columns != x.Columns)
        {
            throw new ArgumentException(
                $"Result panel is {y.Rows}x{y.Columns}, expected {x.Rows}x{x.Columns}.", nameof(y));
        }

        for (int j = 0; j < x.Columns; j++) x.Column(j).CopyTo(y.Column(j));
    }
}
