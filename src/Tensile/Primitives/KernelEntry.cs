namespace Tensile.Primitives;

/// <summary>
/// The one place a view becomes a pointer.
///
/// Everything above this seam handles <see cref="MatrixView{T}"/>s, which carry
/// their extent and are bounds-checked by the runtime. Everything below it is
/// the pointer-and-stride code the kernels were written against. This class is
/// the boundary: it pins each buffer with <c>fixed</c>, calls the primitive,
/// and lets the pin go. No pointer escapes a method here, and no other file in
/// the ergonomic layer may mint one.
///
/// <c>fixed</c> is what makes accepting a span over any memory sound. If the
/// span is over a movable managed array, <c>fixed</c> pins it for the duration
/// of the call and the garbage collector cannot relocate it mid-kernel; if the
/// span is over pinned or native memory, <c>fixed</c> is a no-op. Either way the
/// pointer is valid exactly as long as it exists.
///
/// That guarantee rests on one condition: every primitive called from here is
/// synchronous and returns before the <c>fixed</c> block closes. The parallel
/// GEMM captures pointers into <c>Parallel.For</c> bodies, and that is sound
/// only because <c>Parallel.For</c> blocks until every body has completed. Any
/// future asynchronous path below this seam must pin differently, and this
/// comment is where that constraint is written down.
///
/// The exception is <see cref="FactorLu{TKernel}"/>, which returns a
/// <see cref="LuFactorization"/> that keeps a pointer into <c>a</c> after the
/// pin is released. That is sound only over memory that never moves — a
/// <see cref="Matrix{T}"/>, whose storage is pinned for its lifetime — and so
/// it is internal, called only by <see cref="LuDecomposition"/> over storage it
/// owns. Phase 3 re-plumbs the factorization to hold no pointer and lifts the
/// restriction.
/// </summary>
internal static unsafe class KernelEntry
{
    /// <summary>C := beta*C + alpha*A*B through the dispatch's chosen path.</summary>
    public static void Multiply<TKernel>(
        GemmDispatch dispatch,
        ReadOnlyMatrixView<double> a,
        ReadOnlyMatrixView<double> b,
        MatrixView<double> c,
        double alpha,
        double beta)
        where TKernel : struct, IMicroKernel
    {
        fixed (double* pa = a.Buffer)
        fixed (double* pb = b.Buffer)
        fixed (double* pc = c.Buffer)
        {
            dispatch.Multiply<TKernel>(
                a.Rows, b.Columns, a.Columns,
                alpha, pa, a.Stride,
                pb, b.Stride,
                beta, pc, c.Stride);
        }
    }

    /// <summary>
    /// Factor <paramref name="a"/> in place. The result points into
    /// <paramref name="a"/>'s buffer after this returns, so the caller must
    /// guarantee that buffer never moves — see the class remarks.
    /// </summary>
    public static LuFactorization FactorLu<TKernel>(GemmDispatch dispatch, MatrixView<double> a, int blockSize)
        where TKernel : struct, IMicroKernel
    {
        fixed (double* pa = a.Buffer)
        {
            return Lu.Factor<TKernel>(a.Rows, a.Columns, pa, a.Stride, dispatch, blockSize);
        }
    }

    /// <summary>Solve A*X = B in place against a factorization.</summary>
    public static void SolveLu(LuFactorization lu, MatrixView<double> b)
    {
        fixed (double* pb = b.Buffer)
        {
            Lu.Solve(lu, b.Columns, pb, b.Stride);
        }
    }

    /// <summary>Solve A^T*X = B in place against a factorization.</summary>
    public static void SolveLuTransposed(LuFactorization lu, MatrixView<double> b)
    {
        fixed (double* pb = b.Buffer)
        {
            Lu.SolveTransposed(lu, b.Columns, pb, b.Stride);
        }
    }

    /// <summary>Solve U*X = B in place, U upper triangular with explicit diagonal.</summary>
    public static void SolveUpper(ReadOnlyMatrixView<double> a, MatrixView<double> b)
    {
        fixed (double* pa = a.Buffer)
        fixed (double* pb = b.Buffer)
        {
            Triangular.SolveUpper(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve U^T*X = B in place.</summary>
    public static void SolveUpperTransposed(ReadOnlyMatrixView<double> a, MatrixView<double> b)
    {
        fixed (double* pa = a.Buffer)
        fixed (double* pb = b.Buffer)
        {
            Triangular.SolveUpperTransposed(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L*X = B in place, L lower triangular with explicit diagonal.</summary>
    public static void SolveLower(ReadOnlyMatrixView<double> a, MatrixView<double> b)
    {
        fixed (double* pa = a.Buffer)
        fixed (double* pb = b.Buffer)
        {
            Triangular.SolveLower(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L^T*X = B in place.</summary>
    public static void SolveLowerTransposed(ReadOnlyMatrixView<double> a, MatrixView<double> b)
    {
        fixed (double* pa = a.Buffer)
        fixed (double* pb = b.Buffer)
        {
            Triangular.SolveLowerTransposed(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L*X = B in place, L lower triangular with implicit unit diagonal.</summary>
    public static void SolveLowerUnit(ReadOnlyMatrixView<double> a, MatrixView<double> b)
    {
        fixed (double* pa = a.Buffer)
        fixed (double* pb = b.Buffer)
        {
            Triangular.SolveLowerUnit(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L^T*X = B in place, L with implicit unit diagonal.</summary>
    public static void SolveLowerUnitTransposed(ReadOnlyMatrixView<double> a, MatrixView<double> b)
    {
        fixed (double* pa = a.Buffer)
        fixed (double* pb = b.Buffer)
        {
            Triangular.SolveLowerUnitTransposed(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>||A||_1, the largest absolute column sum.</summary>
    public static double OneNorm(ReadOnlyMatrixView<double> a)
    {
        fixed (double* pa = a.Buffer)
        {
            return Norms.One(a.Rows, a.Columns, pa, a.Stride);
        }
    }

    /// <summary>||A||_inf, the largest absolute row sum.</summary>
    public static double InfinityNorm(ReadOnlyMatrixView<double> a)
    {
        fixed (double* pa = a.Buffer)
        {
            return Norms.Infinity(a.Rows, a.Columns, pa, a.Stride);
        }
    }

    /// <summary>||A||_F, the square root of the sum of squares.</summary>
    public static double FrobeniusNorm(ReadOnlyMatrixView<double> a)
    {
        fixed (double* pa = a.Buffer)
        {
            return Norms.Frobenius(a.Rows, a.Columns, pa, a.Stride);
        }
    }

    /// <summary>
    /// Estimate ||A^power||_1. The estimator holds the pointer across its whole
    /// run, which is sound because the run is synchronous and completes inside
    /// the pin.
    /// </summary>
    public static NormEstimateResult EstimateOneNorm(ReadOnlyMatrixView<double> a, int power, int columns)
    {
        fixed (double* pa = a.Buffer)
        {
            return NormEstimate.OfMatrix(a.Rows, pa, a.Stride, power, columns);
        }
    }
}
