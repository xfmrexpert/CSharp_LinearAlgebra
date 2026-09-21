namespace Tensile.Kernels;

/// <summary>
/// Panel packing. This is the half of BLIS that nothing about the language
/// makes hard, but that everything about performance depends on: the
/// micro-kernel only reaches peak if its operands arrive in contiguous,
/// correctly ordered, aligned buffers.
///
/// Edge panels are zero-padded so the micro-kernel always runs a full MR x NR
/// tile. Only the write-back to C has to care about ragged edges.
///
/// Each panel is independent, so the *Panels variants let several threads pack
/// disjoint ranges of the same buffer with no synchronisation.
/// </summary>
internal static unsafe class Packing
{
    /// <summary>
    /// Pack an mc x kc block of column-major A (column stride lda) into
    /// row-panels of height mr.
    ///
    /// Output layout: panel q occupies [q*mr*kc, (q+1)*mr*kc), and within it
    /// each k contributes mr contiguous doubles.
    /// </summary>
    public static void PackA(int mc, int kc, double* a, int lda, double* ap, int mr) =>
        PackAPanels(0, (mc + mr - 1) / mr, mc, kc, a, lda, ap, mr);

    /// <summary>
    /// Pack a kc x nc block of column-major B (column stride ldb) into
    /// column-panels of width nr.
    ///
    /// Output layout: panel q occupies [q*nr*kc, (q+1)*nr*kc), and within it
    /// each k contributes nr contiguous doubles.
    /// </summary>
    public static void PackB(int kc, int nc, double* b, int ldb, double* bp, int nr) =>
        PackBPanels(0, (nc + nr - 1) / nr, nc, kc, b, ldb, bp, nr);

    /// <summary>
    /// Pack A panels [firstPanel, firstPanel + panelCount). Disjoint ranges may
    /// be packed concurrently into the same <paramref name="ap"/> buffer.
    /// </summary>
    public static void PackAPanels(
        int firstPanel, int panelCount, int mc, int kc, double* a, int lda, double* ap, int mr)
    {
        for (int q = firstPanel; q < firstPanel + panelCount; q++)
        {
            int i0 = q * mr;
            int rows = Math.Min(mr, mc - i0);
            if (rows <= 0) break;

            double* dst = ap + (nint)q * mr * kc;

            for (int p = 0; p < kc; p++)
            {
                // A is column-major, so a whole column slice is contiguous in i.
                double* src = a + (nint)p * lda + i0;

                int i = 0;
                for (; i < rows; i++) dst[i] = src[i];
                for (; i < mr; i++) dst[i] = 0.0;

                dst += mr;
            }
        }
    }

    /// <summary>
    /// Pack B panels [firstPanel, firstPanel + panelCount). Disjoint ranges may
    /// be packed concurrently into the same <paramref name="bp"/> buffer.
    /// </summary>
    public static void PackBPanels(
        int firstPanel, int panelCount, int nc, int kc, double* b, int ldb, double* bp, int nr)
    {
        for (int q = firstPanel; q < firstPanel + panelCount; q++)
        {
            int j0 = q * nr;
            int cols = Math.Min(nr, nc - j0);
            if (cols <= 0) break;

            double* dst = bp + (nint)q * nr * kc;

            for (int p = 0; p < kc; p++)
            {
                // B is column-major, so this gathers across a row: stride ldb.
                int j = 0;
                for (; j < cols; j++) dst[j] = b[(nint)(j0 + j) * ldb + p];
                for (; j < nr; j++) dst[j] = 0.0;

                dst += nr;
            }
        }
    }
}
