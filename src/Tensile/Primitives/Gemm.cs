using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tensile.Primitives;

/// <summary>
/// Reusable packing buffers. These are large (the B buffer is sized to occupy
/// a good fraction of L3), so they are allocated once and reused rather than
/// per call.
/// </summary>
public sealed unsafe class GemmScratch : IDisposable
{
    /// <summary>Row-block size: the packed A block is Mc x Kc and is sized to fit L2.</summary>
    public int Mc { get; }

    /// <summary>Depth of a k-slab, sized so one A and one B micro-panel stay in L1.</summary>
    public int Kc { get; }

    /// <summary>Column-block size: the packed B block is Kc x Nc and is sized to fit L3.</summary>
    public int Nc { get; }

    /// <summary>Packed A block, Mc x Kc, 64-byte aligned.</summary>
    public double* Ap { get; private set; }

    /// <summary>Packed B block, Kc x Nc, 64-byte aligned.</summary>
    public double* Bp { get; private set; }

    /// <summary>Scratch MR x NR tile used to accumulate ragged edges of C.</summary>
    public double* Tile { get; private set; }

    private GemmScratch(int mr, int nr, int mc, int kc, int nc)
    {
        Mc = mc;
        Kc = kc;
        Nc = nc;

        Ap = Alloc((nuint)mc * (nuint)kc);
        Bp = Alloc((nuint)nc * (nuint)kc);
        Tile = Alloc((nuint)mr * (nuint)nr);
    }

    /// <summary>
    /// Cache-blocking parameters. These are placeholders in the spirit of the
    /// BLIS analytical model (Low et al., TOMS 2016): KC is chosen so an A
    /// micro-panel plus a B micro-panel stay resident in L1/L2, MC so the
    /// packed A block fits L2, NC so the packed B block fits L3.
    ///
    /// They are NOT tuned for any particular machine. Sweeping them is the
    /// second experiment, after the micro-kernel question is settled.
    /// </summary>
    public static GemmScratch For<TKernel>() where TKernel : struct, IMicroKernel
    {
        int mr = TKernel.Mr;
        int nr = TKernel.Nr;

        int kc = 384;
        int mc = RoundUp(288, mr);
        int nc = RoundUp(4096, nr);

        return new GemmScratch(mr, nr, mc, kc, nc);
    }

    private static int RoundUp(int value, int multiple) =>
        ((value + multiple - 1) / multiple) * multiple;

    private static double* Alloc(nuint count) =>
        (double*)NativeMemory.AlignedAlloc(count * sizeof(double), 64);

    /// <summary>Release the packing buffers. Safe to call more than once.</summary>
    public void Dispose()
    {
        if (Ap is not null) { NativeMemory.AlignedFree(Ap); Ap = null; }
        if (Bp is not null) { NativeMemory.AlignedFree(Bp); Bp = null; }
        if (Tile is not null) { NativeMemory.AlignedFree(Tile); Tile = null; }
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the native buffers if <see cref="Dispose"/> was not called.</summary>
    ~GemmScratch() => Dispose();
}

/// <summary>
/// The BLIS five-loop GEMM, generic over the micro-kernel. Column-major
/// throughout, no transposes, single-threaded.
///
/// Because TKernel is a struct, the JIT instantiates this method once per
/// kernel type and every TKernel.Execute call is a direct (inlinable) call,
/// not an interface dispatch.
/// </summary>
public static unsafe class Gemm
{
    /// <summary>C := beta*C + alpha*A*B, all column-major.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TKernel>(
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc,
        GemmScratch scratch)
        where TKernel : struct, IMicroKernel
    {
        ScaleC(m, n, beta, c, ldc);

        if (m == 0 || n == 0 || k == 0 || alpha == 0.0) return;

        int mr = TKernel.Mr;
        int nr = TKernel.Nr;
        double* tile = scratch.Tile;

        // Loop 5: partition N into NC-wide column blocks (fills L3 with B).
        for (int jc = 0; jc < n; jc += scratch.Nc)
        {
            int nc = Math.Min(scratch.Nc, n - jc);

            // Loop 4: partition K into KC-deep slabs.
            for (int pc = 0; pc < k; pc += scratch.Kc)
            {
                int kc = Math.Min(scratch.Kc, k - pc);

                Packing.PackB(kc, nc, b + (nint)jc * ldb + pc, ldb, scratch.Bp, nr);

                // Loop 3: partition M into MC-tall row blocks (fills L2 with A).
                for (int ic = 0; ic < m; ic += scratch.Mc)
                {
                    int mc = Math.Min(scratch.Mc, m - ic);

                    Packing.PackA(mc, kc, a + (nint)pc * lda + ic, lda, scratch.Ap, mr);

                    // Loop 2: walk the B micro-panels.
                    for (int jr = 0; jr < nc; jr += nr)
                    {
                        int nrb = Math.Min(nr, nc - jr);
                        double* bPanel = scratch.Bp + (nint)(jr / nr) * nr * kc;

                        // Loop 1: walk the A micro-panels. Body is the kernel.
                        for (int ir = 0; ir < mc; ir += mr)
                        {
                            int mrb = Math.Min(mr, mc - ir);
                            double* aPanel = scratch.Ap + (nint)(ir / mr) * mr * kc;
                            double* cTile = c + (nint)(jc + jr) * ldc + (ic + ir);

                            if (mrb == mr && nrb == nr)
                            {
                                TKernel.Execute(kc, alpha, aPanel, bPanel, cTile, ldc);
                            }
                            else
                            {
                                // Ragged edge: accumulate into a full tile, then
                                // scatter back only the rows and columns that exist.
                                new Span<double>(tile, mr * nr).Clear();
                                TKernel.Execute(kc, alpha, aPanel, bPanel, tile, mr);

                                for (int j = 0; j < nrb; j++)
                                    for (int i = 0; i < mrb; i++)
                                        cTile[(nint)j * ldc + i] += tile[j * mr + i];
                            }
                        }
                    }
                }
            }
        }
    }

    private static void ScaleC(int m, int n, double beta, double* c, int ldc)
    {
        if (beta == 1.0) return;

        if (beta == 0.0)
        {
            for (int j = 0; j < n; j++)
                new Span<double>(c + (nint)j * ldc, m).Clear();
            return;
        }

        for (int j = 0; j < n; j++)
        {
            double* col = c + (nint)j * ldc;
            for (int i = 0; i < m; i++) col[i] *= beta;
        }
    }
}
