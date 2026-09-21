using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tensile.Kernels;

/// <summary>
/// Buffers and blocking parameters for <see cref="ParallelGemm"/>.
///
/// Unlike the single-threaded scratch, A is packed for the whole m x KC slab in
/// one go rather than one MC block at a time. That turns A packing into a
/// single parallel phase with no barriers inside it, and costs nothing in
/// locality: the compute phase still walks MC blocks in order, so every thread
/// is reading the same A block at the same time and it stays resident in each
/// core's L2.
/// </summary>
internal sealed unsafe class ParallelGemmScratch : IDisposable
{
    /// <summary>Row-block size: the packed A block per MC step is sized to fit a core's share of L2.</summary>
    public int Mc { get; set; }

    /// <summary>Depth of a k-slab, sized against the smallest L1 in the machine.</summary>
    public int Kc { get; set; }

    /// <summary>Column-block size: the packed B block is sized to sit in L3.</summary>
    public int Nc { get; set; }

    /// <summary>Upper bound on worker threads. The actual count is scaled down for small problems.</summary>
    public int MaxThreads { get; set; }

    /// <summary>Packed A slab for the whole m x Kc extent, 64-byte aligned.</summary>
    public double* Ap { get; private set; }

    /// <summary>Packed B block, Kc x Nc, 64-byte aligned.</summary>
    public double* Bp { get; private set; }

    private nuint _aCapacity;
    private nuint _bCapacity;

    private ParallelGemmScratch(int mc, int kc, int nc, int threads)
    {
        Mc = mc;
        Kc = kc;
        Nc = nc;
        MaxThreads = threads;
    }

    /// <summary>
    /// Blocking parameters for a multi-core run.
    ///
    /// KC is chosen so one A micro-panel plus one B micro-panel fit in the
    /// *smallest* L1 in the machine — 32 KiB on Gracemont E-cores, versus 48 KiB
    /// on Golden Cove P-cores. At KC=256 with MR=8/NR=6 that is 16 KiB + 12 KiB.
    ///
    /// MC is chosen so the packed A block fits the smallest per-core share of
    /// L2. On a 12700H that is an E-core's quarter of its cluster's 2 MiB, so
    /// 512 KiB; MC=144 gives a 288 KiB block, leaving headroom.
    ///
    /// NC sizes the packed B block to sit in L3 alongside everything else.
    ///
    /// These are starting points derived from cache geometry, not tuned values.
    /// Sweep them.
    /// </summary>
    public static ParallelGemmScratch For<TKernel>(int threads = 0)
        where TKernel : struct, IMicroKernel
    {
        int mr = TKernel.Mr;
        int nr = TKernel.Nr;

        if (threads <= 0) threads = Environment.ProcessorCount;

        return new ParallelGemmScratch(
            mc: RoundUp(144, mr),
            kc: 256,
            nc: RoundUp(4096, nr),
            threads: threads);
    }

    internal void Ensure(int m, int nc, int kc, int mr, int nr)
    {
        nuint aNeeded = (nuint)(((m + mr - 1) / mr) * mr) * (nuint)kc;
        nuint bNeeded = (nuint)(((nc + nr - 1) / nr) * nr) * (nuint)kc;

        if (aNeeded > _aCapacity)
        {
            if (Ap is not null) NativeMemory.AlignedFree(Ap);
            Ap = Alloc(aNeeded);
            _aCapacity = aNeeded;
        }

        if (bNeeded > _bCapacity)
        {
            if (Bp is not null) NativeMemory.AlignedFree(Bp);
            Bp = Alloc(bNeeded);
            _bCapacity = bNeeded;
        }
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
        _aCapacity = 0;
        _bCapacity = 0;
        GC.SuppressFinalize(this);
    }

    /// <summary>Releases the native buffers if <see cref="Dispose"/> was not called.</summary>
    ~ParallelGemmScratch() => Dispose();
}

/// <summary>
/// Multi-threaded blocked GEMM.
///
/// Parallelism is over loop 2 (the jr micro-panel loop), not loop 3. That
/// matters on a hybrid CPU: loop 3 would give only ceil(m/MC) work items — 15
/// at m=2048 — which is fewer than the machine has threads, so the schedule
/// would be dominated by whichever block landed on an E-core. Loop 2 gives
/// ceil(nc/NR) items instead (341 at n=2048), small enough that .NET's dynamic
/// partitioner balances P- and E-cores by itself, with no core-type detection,
/// no affinity calls, and automatic adaptation to thermal throttling.
///
/// Each phase is a separate Parallel.For, which is an implicit barrier. There
/// are 2 + ceil(m/MC) of them per k-slab. At n=2048 that is roughly 3% of
/// runtime; if it shows up as poor scaling at small n, the fix is persistent
/// worker threads with an explicit Barrier rather than the thread pool.
/// </summary>
internal static unsafe class ParallelGemm
{
    /// <summary>C := beta*C + alpha*A*B, all column-major.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TKernel>(
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc,
        ParallelGemmScratch scratch)
        where TKernel : struct, IMicroKernel
    {
        long work = (long)m * n * k;
        int threads = Math.Clamp((int)(work / 8_000_000), 1, scratch.MaxThreads);
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };

        ScaleC(m, n, beta, c, ldc, options);

        if (m == 0 || n == 0 || k == 0 || alpha == 0.0) return;

        int mr = TKernel.Mr;
        int nr = TKernel.Nr;

        // Pointers cannot be captured by a lambda, so they travel as nint.
        nint cAddr = (nint)c;

        for (int jc = 0; jc < n; jc += scratch.Nc)
        {
            int nc = Math.Min(scratch.Nc, n - jc);
            int jcBase = jc;

            for (int pc = 0; pc < k; pc += scratch.Kc)
            {
                int kc = Math.Min(scratch.Kc, k - pc);

                scratch.Ensure(m, nc, kc, mr, nr);

                nint apAddr = (nint)scratch.Ap;
                nint bpAddr = (nint)scratch.Bp;
                nint aAddr = (nint)(a + (nint)pc * lda);
                nint bAddr = (nint)(b + (nint)jc * ldb + pc);

                int bPanels = (nc + nr - 1) / nr;
                int aPanels = (m + mr - 1) / mr;

                // Phase 1: pack the B block. Panels are disjoint, so no locking.
                Parallel.For(0, bPanels, options, q =>
                    Packing.PackBPanels(q, 1, nc, kc, (double*)bAddr, ldb, (double*)bpAddr, nr));

                // Phase 2: pack the whole A slab at once.
                Parallel.For(0, aPanels, options, q =>
                    Packing.PackAPanels(q, 1, m, kc, (double*)aAddr, lda, (double*)apAddr, mr));

                // Phase 3: walk MC blocks in order so all threads share one A
                // block; within each, distribute the B micro-panels dynamically.
                for (int ic = 0; ic < m; ic += scratch.Mc)
                {
                    int mc = Math.Min(scratch.Mc, m - ic);
                    int icBase = ic;
                    int firstAPanel = ic / mr;
                    int aPanelCount = (mc + mr - 1) / mr;

                    Parallel.For(0, bPanels, options, q =>
                    {
                        int jr = q * nr;
                        int nrb = Math.Min(nr, nc - jr);
                        double* bPanel = (double*)bpAddr + (nint)q * nr * kc;

                        double* tile = stackalloc double[mr * nr];

                        for (int t = 0; t < aPanelCount; t++)
                        {
                            int ir = t * mr;
                            int mrb = Math.Min(mr, mc - ir);
                            double* aPanel = (double*)apAddr + (nint)(firstAPanel + t) * mr * kc;
                            double* cTile = (double*)cAddr + (nint)(jcBase + jr) * ldc + (icBase + ir);

                            if (mrb == mr && nrb == nr)
                            {
                                TKernel.Execute(kc, alpha, aPanel, bPanel, cTile, ldc);
                            }
                            else
                            {
                                new Span<double>(tile, mr * nr).Clear();
                                TKernel.Execute(kc, alpha, aPanel, bPanel, tile, mr);

                                for (int j = 0; j < nrb; j++)
                                    for (int i = 0; i < mrb; i++)
                                        cTile[(nint)j * ldc + i] += tile[j * mr + i];
                            }
                        }
                    });
                }
            }
        }
    }

    private static void ScaleC(int m, int n, double beta, double* c, int ldc, ParallelOptions options)
    {
        if (beta == 1.0) return;

        nint cAddr = (nint)c;

        if (beta == 0.0)
        {
            Parallel.For(0, n, options, j =>
                new Span<double>((double*)cAddr + (nint)j * ldc, m).Clear());
            return;
        }

        Parallel.For(0, n, options, j =>
        {
            double* col = (double*)cAddr + (nint)j * ldc;
            for (int i = 0; i < m; i++) col[i] *= beta;
        });
    }
}
