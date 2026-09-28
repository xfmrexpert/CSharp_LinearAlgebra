namespace Tensile;

/// <summary>
/// The six real buffers a 4M complex product works in -- real and imaginary
/// parts of A, B and the product -- kept by a <see cref="Workspace"/> so that
/// repeated products reuse them instead of allocating per call.
///
/// Unlike the packing buffers, which are fixed by the block sizes, these are
/// sized by the problem: O(mk + kn + mn) doubles. A workspace that simply kept
/// the largest it had ever seen would hold ~48 MB forever after one 1000x1000
/// product, and <see cref="Workspace.Shared"/> lives as long as the process.
/// So retention is capped at <see cref="RetainedLimit"/> elements in total:
/// a product whose buffers fit is served from retained storage, growing it as
/// needed, and one that does not gets buffers of its own for that call only,
/// which is exactly what every product did before this existed. The large
/// products are the ones that go back to allocating, and they are also the
/// ones whose O(n^3) of work amortises it best.
///
/// Nothing needs zeroing on reuse, and that is a contract of the caller, not
/// of this type: the split writes every element of the four operand buffers,
/// and the first product into each part of the result runs with beta = 0,
/// which overwrites. Stale contents from an earlier product are never read.
///
/// Not thread-safe. The workspace hands it out only under its lock, and the
/// buffers it returns must not outlive that lock: while one product holds
/// them, the next caller would otherwise be writing into them.
/// </summary>
internal sealed class ComplexScratch
{
    /// <summary>
    /// The default cap on retained elements: 2^21 doubles, 16 MiB. That is a
    /// 512x512 complex product in full (six buffers of 512^2), which covers
    /// the target application's frequency-domain matrices, whose order is a
    /// few hundred.
    /// </summary>
    internal const long DefaultRetainedLimit = 1L << 21;

    private const int SlotCount = 6;

    private static readonly string[] Purposes =
    [
        "the real part of A in a complex product",
        "the imaginary part of A in a complex product",
        "the real part of B in a complex product",
        "the imaginary part of B in a complex product",
        "the real part of a complex product",
        "the imaginary part of a complex product",
    ];

    private readonly double[][] _slots = [[], [], [], [], [], []];

    /// <summary>Create scratch that retains at most <paramref name="retainedLimit"/> elements.</summary>
    internal ComplexScratch(long retainedLimit = DefaultRetainedLimit)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(retainedLimit);
        RetainedLimit = retainedLimit;
    }

    /// <summary>The most elements this scratch will keep between products.</summary>
    internal long RetainedLimit { get; }

    /// <summary>Elements currently kept between products, for the tests.</summary>
    internal long RetainedElements
    {
        get
        {
            long total = 0;
            foreach (double[] slot in _slots) total += slot.Length;
            return total;
        }
    }

    /// <summary>
    /// Buffers for an m x k by k x n product: retained if they fit under the
    /// cap, otherwise allocated for this call alone.
    ///
    /// Retained storage only grows while the running total stays under the
    /// cap. When a product fits the cap but growing to cover it would not --
    /// a workload alternating between tall and wide shapes, say -- the slots
    /// are re-cut to exactly this product, so the cap bounds what is kept
    /// rather than what has ever been seen.
    /// </summary>
    /// <exception cref="AllocationLimitException">A buffer exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    internal SplitBuffers Acquire(int m, int k, int n)
    {
        // Each fits int: A, B and C are views that already exist, and a view's
        // extent is at least rows*columns.
        long mk = (long)m * k, kn = (long)k * n, mn = (long)m * n;
        ReadOnlySpan<long> required = [mk, mk, kn, kn, mn, mn];

        long grown = 0, exact = 0;
        for (int i = 0; i < SlotCount; i++)
        {
            grown += Math.Max(_slots[i].Length, required[i]);
            exact += required[i];
        }

        if (grown <= RetainedLimit)
        {
            for (int i = 0; i < SlotCount; i++)
            {
                if (_slots[i].Length < required[i]) Replace(i, required[i]);
            }

            return new SplitBuffers(_slots);
        }

        if (exact <= RetainedLimit)
        {
            for (int i = 0; i < SlotCount; i++)
            {
                if (_slots[i].Length != required[i]) Replace(i, required[i]);
            }

            return new SplitBuffers(_slots);
        }

        var transient = new double[SlotCount][];
        for (int i = 0; i < SlotCount; i++) transient[i] = Storage.Array<double>((int)required[i], Purposes[i]);

        return new SplitBuffers(transient);
    }

    /// <summary>Drop everything retained. The next product starts from nothing.</summary>
    internal void Release()
    {
        for (int i = 0; i < SlotCount; i++) _slots[i] = [];
    }

    private void Replace(int slot, long elements)
    {
        // Let go of the old buffer before asking for the new one, so that a
        // growth never needs both at once, and a refused allocation leaves an
        // empty slot rather than a stale undersized one.
        _slots[slot] = [];
        _slots[slot] = Storage.Array<double>((int)elements, Purposes[slot]);
    }
}

/// <summary>
/// The buffers for one 4M product, bound to exact shapes on request. A value
/// type over six array references, so handing a set out allocates nothing.
/// </summary>
internal readonly struct SplitBuffers
{
    private readonly double[] _aRe, _aIm, _bRe, _bIm, _re, _im;

    internal SplitBuffers(double[][] slots)
    {
        (_aRe, _aIm, _bRe, _bIm, _re, _im) = (slots[0], slots[1], slots[2], slots[3], slots[4], slots[5]);
    }

    internal MatrixView<double> RealA(int rows, int columns) => Bind(_aRe, rows, columns);

    internal MatrixView<double> ImaginaryA(int rows, int columns) => Bind(_aIm, rows, columns);

    internal MatrixView<double> RealB(int rows, int columns) => Bind(_bRe, rows, columns);

    internal MatrixView<double> ImaginaryB(int rows, int columns) => Bind(_bIm, rows, columns);

    internal MatrixView<double> RealProduct(int rows, int columns) => Bind(_re, rows, columns);

    internal MatrixView<double> ImaginaryProduct(int rows, int columns) => Bind(_im, rows, columns);

    // Packed, unit stride: a retained buffer may be longer than this product
    // needs, and Bind takes exactly the shape's extent from the front of it.
    private static MatrixView<double> Bind(double[] buffer, int rows, int columns) =>
        MatrixView<double>.Bind(buffer, new MatrixShape(rows, columns));
}
