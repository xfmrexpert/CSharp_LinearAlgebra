using System.Numerics;
using System.Runtime.CompilerServices;

namespace Tensile.Kernels;

/// <summary>
/// Pivot selection for LU.
///
/// This is a search rather than an arithmetic update, which is why it sits
/// apart from <see cref="ColumnOps"/> despite sharing their shape: what has to
/// be right about it is the index it returns, not a residual. Its only caller
/// is the unblocked panel factorization in <see cref="Lu"/>.
/// </summary>
internal static unsafe class Pivoting
{
    /// <summary>
    /// Index of the element with largest magnitude, ties resolving to the
    /// lowest index. Returns 0 for n &lt;= 0.
    ///
    /// The tie rule is not cosmetic: it is what makes the pivot sequence
    /// reproducible, and it matches LAPACK's idamax, so a factorization can be
    /// compared against one produced there.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int IndexOfMaxAbs(int n, double* x)
    {
        if (n <= 0) return 0;

        int width = Vector<double>.Count;
        int best = 0;
        double bestValue = Math.Abs(x[0]);
        int i = 1;

        // Below two vector widths the reduction costs more than it saves.
        if (Vector.IsHardwareAccelerated && n >= 2 * width)
        {
            Span<long> lanes = stackalloc long[width];
            for (int lane = 0; lane < width; lane++) lanes[lane] = lane;

            var indices = new Vector<long>(lanes);
            var step = new Vector<long>((long)width);

            var bestVector = Vector.Abs(new Vector<double>(new ReadOnlySpan<double>(x, width)));
            var bestIndices = indices;

            // Strict greater-than keeps the earliest index within each lane;
            // the reduction below breaks cross-lane ties the same way. Together
            // that reproduces the scalar scan exactly, NaNs included: every
            // comparison against a NaN is false in both forms.
            for (i = width; i <= n - width; i += width)
            {
                indices += step;

                var value = Vector.Abs(new Vector<double>(new ReadOnlySpan<double>(x + i, width)));
                var greater = Vector.GreaterThan(value, bestVector);

                bestVector = Vector.ConditionalSelect(greater, value, bestVector);
                bestIndices = Vector.ConditionalSelect(
                    Vector.AsVectorInt64(greater), indices, bestIndices);
            }

            bestValue = bestVector[0];
            best = (int)bestIndices[0];

            for (int lane = 1; lane < width; lane++)
            {
                double value = bestVector[lane];
                int index = (int)bestIndices[lane];

                if (value > bestValue || (value == bestValue && index < best))
                {
                    bestValue = value;
                    best = index;
                }
            }
        }

        for (; i < n; i++)
        {
            double value = Math.Abs(x[i]);
            if (value > bestValue)
            {
                bestValue = value;
                best = i;
            }
        }

        return best;
    }
}
