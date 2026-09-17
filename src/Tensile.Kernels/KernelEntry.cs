namespace Tensile.Kernels;

/// <summary>
/// The one place a span becomes a pointer.
///
/// Everything above this seam handles spans, which carry their length and are
/// bounds-checked by the runtime. Everything below it is the pointer-and-stride
/// code the kernels were written against. This class is the boundary: it pins
/// each operand with <c>fixed</c>, calls the primitive, and lets the pin go. No
/// pointer escapes a method here, and the public assembly cannot mint one at
/// all -- it compiles with <c>AllowUnsafeBlocks</c> off.
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
/// Nothing returned from here holds a pointer. <see cref="FactorLu{TKernel}"/>
/// returns a <see cref="LuFactorization"/> that carries the pivots and
/// diagnostics only, and the solves take the factors back as an operand, so a
/// caller may factor a view over any memory and pin it only for each call.
/// </summary>
internal static unsafe class KernelEntry
{
    /// <summary>C := beta*C + alpha*A*B through the dispatch's chosen path.</summary>
    public static void Multiply<TKernel>(
        GemmDispatch dispatch, Operand a, Operand b, Target c, double alpha, double beta)
        where TKernel : struct, IMicroKernel
    {
        Require(a.Columns == b.Rows, "inner dimensions");
        Require(c.Rows == a.Rows && c.Columns == b.Columns, "destination shape");

        // A destination with no elements has nothing to receive, however many
        // columns it nominally has. (An empty inner dimension is different: C
        // is real and still has to be scaled by beta, so it goes through.)
        if (c.Rows == 0 || c.Columns == 0) return;

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        fixed (double* pc = c.Data)
        {
            dispatch.Multiply<TKernel>(
                a.Rows, b.Columns, a.Columns,
                alpha, pa, a.Stride,
                pb, b.Stride,
                beta, pc, c.Stride);
        }
    }

    /// <summary>Factor <paramref name="a"/> in place as P*A = L*U. The result holds no pointer.</summary>
    public static LuFactorization FactorLu<TKernel>(GemmDispatch dispatch, Target a, int blockSize)
        where TKernel : struct, IMicroKernel
    {
        fixed (double* pa = a.Data)
        {
            return Lu.Factor<TKernel>(a.Rows, a.Columns, pa, a.Stride, dispatch, blockSize);
        }
    }

    /// <summary>Solve A*X = B in place against a factorization, the packed factors supplied as an operand.</summary>
    public static void SolveLu(LuFactorization lu, Operand factors, Target b)
    {
        RequireFactors(lu, factors);
        Require(b.Rows == lu.Rows, "right-hand side rows");
        if (IsEmpty(b)) return;

        fixed (double* pf = factors.Data)
        fixed (double* pb = b.Data)
        {
            Lu.Solve(lu, pf, factors.Stride, b.Columns, pb, b.Stride);
        }
    }

    /// <summary>Solve A^T*X = B in place against a factorization.</summary>
    public static void SolveLuTransposed(LuFactorization lu, Operand factors, Target b)
    {
        RequireFactors(lu, factors);
        Require(b.Rows == lu.Rows, "right-hand side rows");
        if (IsEmpty(b)) return;

        fixed (double* pf = factors.Data)
        fixed (double* pb = b.Data)
        {
            Lu.SolveTransposed(lu, pf, factors.Stride, b.Columns, pb, b.Stride);
        }
    }

    /// <summary>Solve U*X = B in place, U upper triangular with explicit diagonal.</summary>
    public static void SolveUpper(Operand a, Target b)
    {
        RequireTriangular(a, b);
        if (IsEmpty(b)) return;

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        {
            Triangular.SolveUpper(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve U^T*X = B in place.</summary>
    public static void SolveUpperTransposed(Operand a, Target b)
    {
        RequireTriangular(a, b);
        if (IsEmpty(b)) return;

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        {
            Triangular.SolveUpperTransposed(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L*X = B in place, L lower triangular with explicit diagonal.</summary>
    public static void SolveLower(Operand a, Target b)
    {
        RequireTriangular(a, b);
        if (IsEmpty(b)) return;

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        {
            Triangular.SolveLower(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L^T*X = B in place.</summary>
    public static void SolveLowerTransposed(Operand a, Target b)
    {
        RequireTriangular(a, b);
        if (IsEmpty(b)) return;

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        {
            Triangular.SolveLowerTransposed(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L*X = B in place, L lower triangular with implicit unit diagonal.</summary>
    public static void SolveLowerUnit(Operand a, Target b)
    {
        RequireTriangular(a, b);
        if (IsEmpty(b)) return;

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        {
            Triangular.SolveLowerUnit(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Solve L^T*X = B in place, L with implicit unit diagonal.</summary>
    public static void SolveLowerUnitTransposed(Operand a, Target b)
    {
        RequireTriangular(a, b);
        if (IsEmpty(b)) return;

        fixed (double* pa = a.Data)
        fixed (double* pb = b.Data)
        {
            Triangular.SolveLowerUnitTransposed(a.Rows, b.Columns, pa, a.Stride, pb, b.Stride);
        }
    }

    /// <summary>Y := A*X for square A and a panel X of matching row count.</summary>
    public static void MultiplyPanel(Operand a, Operand x, Target y)
    {
        RequirePanel(a, x, y);
        if (IsEmpty(y)) return;

        fixed (double* pa = a.Data)
        fixed (double* px = x.Data)
        fixed (double* py = y.Data)
        {
            Blas2.Multiply(a.Rows, x.Columns, pa, a.Stride, px, x.Stride, py, y.Stride);
        }
    }

    /// <summary>Y := A^T*X for square A and a panel X of matching row count.</summary>
    public static void MultiplyPanelTransposed(Operand a, Operand x, Target y)
    {
        RequirePanel(a, x, y);
        if (IsEmpty(y)) return;

        fixed (double* pa = a.Data)
        fixed (double* px = x.Data)
        fixed (double* py = y.Data)
        {
            Blas2.MultiplyTransposed(a.Rows, x.Columns, pa, a.Stride, px, x.Stride, py, y.Stride);
        }
    }

    /// <summary>||A||_1, the largest absolute column sum.</summary>
    public static double OneNorm(Operand a)
    {
        if (IsEmpty(a)) return 0.0;

        fixed (double* pa = a.Data)
        {
            return Norms.One(a.Rows, a.Columns, pa, a.Stride);
        }
    }

    /// <summary>||A||_inf, the largest absolute row sum.</summary>
    public static double InfinityNorm(Operand a)
    {
        if (IsEmpty(a)) return 0.0;

        fixed (double* pa = a.Data)
        {
            return Norms.Infinity(a.Rows, a.Columns, pa, a.Stride);
        }
    }

    /// <summary>||A||_F, the square root of the sum of squares.</summary>
    public static double FrobeniusNorm(Operand a)
    {
        if (IsEmpty(a)) return 0.0;

        fixed (double* pa = a.Data)
        {
            return Norms.Frobenius(a.Rows, a.Columns, pa, a.Stride);
        }
    }

    // The public assembly validates shapes before it gets here and reports
    // them as argument errors. These are the seam's own preconditions,
    // restated so that a primitive can never be reached with operands that
    // disagree, whatever the caller above did.

    /// <summary>
    /// No elements, whatever the nominal column count. The kernels walk
    /// columns, and a 0 x 2^31 operand would have them walk two billion
    /// times over nothing; every entry point returns before that happens.
    /// </summary>
    private static bool IsEmpty(Operand a) => a.Rows == 0 || a.Columns == 0;

    private static bool IsEmpty(Target a) => a.Rows == 0 || a.Columns == 0;

    private static void Require(bool condition, string what)
    {
        if (!condition) throw new ArgumentException($"Kernel operands disagree on {what}.");
    }

    private static void RequireFactors(LuFactorization lu, Operand factors) =>
        Require(factors.Rows == lu.Rows && factors.Columns == lu.Columns, "factor shape");

    private static void RequireTriangular(Operand a, Target b)
    {
        Require(a.Rows == a.Columns, "triangular operand squareness");
        Require(b.Rows == a.Rows, "right-hand side rows");
    }

    private static void RequirePanel(Operand a, Operand x, Target y)
    {
        Require(a.Rows == a.Columns, "operator squareness");
        Require(x.Rows == a.Rows && y.Rows == a.Rows, "panel rows");
        Require(x.Columns == y.Columns, "panel columns");
    }
}
