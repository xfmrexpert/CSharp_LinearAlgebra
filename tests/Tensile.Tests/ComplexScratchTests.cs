using System.Numerics;

namespace Tensile.Tests;

/// <summary>
/// The split buffers a workspace keeps for 4M products: the retention policy
/// on its own, then what reuse must not change about the products themselves.
///
/// Reuse is observed through <see cref="MatrixView{T}.Overlaps"/>: two views
/// over the same retained buffer overlap, and two over separate allocations
/// cannot.
/// </summary>
public class ComplexScratchTests
{
    // ---- the policy ----------------------------------------------------------

    [Fact]
    public void AProductThatFitsIsRetained()
    {
        var scratch = new ComplexScratch(retainedLimit: 1000);

        _ = scratch.Acquire(4, 5, 6);

        // Two parts each of A (4x5), B (5x6) and the product (4x6).
        Assert.Equal(2 * (20 + 30 + 24), scratch.RetainedElements);
    }

    [Fact]
    public void RetainedBuffersAreReused()
    {
        var scratch = new ComplexScratch(retainedLimit: 1000);

        SplitBuffers first = scratch.Acquire(4, 5, 6);
        SplitBuffers second = scratch.Acquire(4, 5, 6);

        Assert.True(first.RealA(4, 5).Overlaps(second.RealA(4, 5)));
        Assert.True(first.ImaginaryProduct(4, 6).Overlaps(second.ImaginaryProduct(4, 6)));
    }

    /// <summary>
    /// A smaller product reuses the front of the larger buffers, which is the
    /// shape of an LU's trailing updates: the first is the largest.
    /// </summary>
    [Fact]
    public void ASmallerProductReusesAndDoesNotShrink()
    {
        var scratch = new ComplexScratch(retainedLimit: 1000);

        SplitBuffers large = scratch.Acquire(10, 10, 10);
        SplitBuffers small = scratch.Acquire(3, 2, 4);

        Assert.Equal(600, scratch.RetainedElements);
        Assert.True(large.RealB(10, 10).Overlaps(small.RealB(2, 4)));
    }

    [Fact]
    public void AProductOverTheCapGetsBuffersOfItsOwn()
    {
        var scratch = new ComplexScratch(retainedLimit: 100);

        SplitBuffers first = scratch.Acquire(10, 10, 10);
        SplitBuffers second = scratch.Acquire(10, 10, 10);

        Assert.Equal(0, scratch.RetainedElements);
        Assert.False(first.RealA(10, 10).Overlaps(second.RealA(10, 10)));
    }

    /// <summary>
    /// A product over the cap leaves what is retained alone, so the next
    /// product that fits still finds it.
    /// </summary>
    [Fact]
    public void AnOversizedProductDoesNotDisturbRetainedBuffers()
    {
        var scratch = new ComplexScratch(retainedLimit: 200);

        SplitBuffers before = scratch.Acquire(4, 4, 4);
        _ = scratch.Acquire(20, 20, 20);
        SplitBuffers after = scratch.Acquire(4, 4, 4);

        Assert.Equal(96, scratch.RetainedElements);
        Assert.True(before.RealA(4, 4).Overlaps(after.RealA(4, 4)));
    }

    /// <summary>
    /// Tall then wide: each fits the cap, but growing every buffer to cover
    /// both would not, so the buffers are re-cut to the product in hand. The
    /// cap bounds what is kept, not what has ever been asked for.
    /// </summary>
    [Fact]
    public void GrowthThatWouldPassTheCapReCutsToTheProduct()
    {
        var scratch = new ComplexScratch(retainedLimit: 300);

        _ = scratch.Acquire(10, 10, 1);             // A 100, B 10, C 10: 240 retained
        Assert.Equal(240, scratch.RetainedElements);

        _ = scratch.Acquire(1, 10, 10);             // A 10, B 100, C 10: growing to both would be 420
        Assert.Equal(240, scratch.RetainedElements);
        Assert.True(scratch.RetainedElements <= scratch.RetainedLimit);
    }

    [Fact]
    public void ReleaseDropsEverything()
    {
        var scratch = new ComplexScratch(retainedLimit: 1000);

        _ = scratch.Acquire(5, 5, 5);
        scratch.Release();

        Assert.Equal(0, scratch.RetainedElements);
    }

    [Fact]
    public void TheDefaultCapCoversAProductOfOrder512()
    {
        var scratch = new ComplexScratch();

        Assert.Equal(1L << 21, scratch.RetainedLimit);
        Assert.True(6L * 512 * 512 <= scratch.RetainedLimit);
    }

    // ---- what reuse must not change -----------------------------------------

    [Fact]
    public void AWorkspaceRetainsTheBuffersOfAProduct()
    {
        using var workspace = new Workspace(multithreaded: false);

        var a = new Matrix<Complex>(7, 5);
        var b = new Matrix<Complex>(5, 3);
        var c = new Matrix<Complex>(7, 3);

        workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View);

        Assert.Equal(2 * (35 + 15 + 21), RetainedBy(workspace));
    }

    /// <summary>
    /// A previous, larger product leaves NaN in every buffer. The next product
    /// must not read any of it: Split overwrites the operand parts and the
    /// first product into each result part overwrites with beta = 0.
    /// </summary>
    [Fact]
    public void AStaleBufferIsNeverRead()
    {
        using var workspace = new Workspace(multithreaded: false);

        var poisonA = new Matrix<Complex>(20, 20);
        var poisonB = new Matrix<Complex>(20, 20);
        poisonA.Fill(new Complex(double.NaN, double.NaN));
        poisonB.Fill(new Complex(double.NaN, double.NaN));
        workspace.Multiply(poisonA.ReadOnlyView, poisonB.ReadOnlyView, new Matrix<Complex>(20, 20).View);

        var a = new Matrix<Complex>(7, 3);
        var b = new Matrix<Complex>(3, 5);
        var c = new Matrix<Complex>(7, 5);
        ComplexMultiplyContract<ScalarCase>.Fill(a, seed: 21);
        ComplexMultiplyContract<ScalarCase>.Fill(b, seed: 22);

        workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View);

        ComplexMultiplyContract<ScalarCase>.AssertWithinBound(
            c, a, b, ComplexMultiplyContract<ScalarCase>.Reference(a, b, null, Complex.One, Complex.Zero));
    }

    /// <summary>
    /// A 1 x k by k x 1 product with k large enough that its split buffers
    /// exceed the default cap -- cheap to compute, expensive to keep. It is
    /// correct, and the workspace keeps none of it.
    /// </summary>
    [Fact]
    public void AProductOverTheDefaultCapIsCorrectAndNotRetained()
    {
        const int K = 600_000;                      // 2*(2K + 1) > 2^21

        using var workspace = new Workspace(multithreaded: false);

        var a = new Matrix<Complex>(1, K);
        var b = new Matrix<Complex>(K, 1);
        var c = new Matrix<Complex>(1, 1);
        ComplexMultiplyContract<ScalarCase>.Fill(a, seed: 23);
        ComplexMultiplyContract<ScalarCase>.Fill(b, seed: 24);

        workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View);

        ComplexMultiplyContract<ScalarCase>.AssertWithinBound(
            c, a, b, ComplexMultiplyContract<ScalarCase>.Reference(a, b, null, Complex.One, Complex.Zero));
        Assert.Equal(0, RetainedBy(workspace));
    }

    /// <summary>
    /// Many callers, one workspace, products of different shapes. The lock
    /// has to cover the whole of each product -- split, four GEMMs, combine --
    /// or one caller's parts would land in another's buffers between steps.
    /// </summary>
    [Fact]
    public async Task ConcurrentProductsOnOneWorkspaceDoNotInterfere()
    {
        using var workspace = new Workspace(multithreaded: false);

        var tasks = Enumerable.Range(0, 8).Select(worker => Task.Run(() =>
        {
            for (int round = 0; round < 12; round++)
            {
                int m = 3 + ((worker * 5) + round) % 17, k = 2 + ((worker * 3) + (round * 7)) % 13, n = 1 + (worker + round) % 11;

                var a = new Matrix<Complex>(m, k);
                var b = new Matrix<Complex>(k, n);
                var c = new Matrix<Complex>(m, n);
                ComplexMultiplyContract<ScalarCase>.Fill(a, seed: (worker * 100) + round);
                ComplexMultiplyContract<ScalarCase>.Fill(b, seed: (worker * 100) + round + 50);

                workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View);

                ComplexMultiplyContract<ScalarCase>.AssertWithinBound(
                    c, a, b, ComplexMultiplyContract<ScalarCase>.Reference(a, b, null, Complex.One, Complex.Zero));
            }
        }, TestContext.Current.CancellationToken));

        await Task.WhenAll(tasks);
    }

    [Fact]
    public void DisposingReleasesTheScratch()
    {
        var workspace = new Workspace(multithreaded: false);
        workspace.Multiply(new Matrix<Complex>(4, 4).ReadOnlyView, new Matrix<Complex>(4, 4).ReadOnlyView, new Matrix<Complex>(4, 4).View);

        workspace.Dispose();

        lock (workspace.Gate)
        {
            Assert.Throws<ObjectDisposedException>(() => workspace.ComplexScratch);
        }
    }

    internal static long RetainedBy(Workspace workspace)
    {
        lock (workspace.Gate)
        {
            return workspace.ComplexScratch.RetainedElements;
        }
    }
}

/// <summary>
/// The split buffers under the allocation limit. The limit governs allocation,
/// so a buffer the workspace would have to allocate is refused -- before C is
/// touched -- while one it already retains is not an allocation at all.
/// </summary>
[Collection(nameof(Invariants.TensileLimitsCollection))]
public sealed class ComplexScratchLimitTests : IDisposable
{
    public void Dispose() => TensileLimits.Reset();

    [Fact]
    public void ARefusedSplitBufferLeavesTheDestinationUntouched()
    {
        var a = new Matrix<Complex>(30, 30);
        var b = new Matrix<Complex>(30, 30);
        var c = new Matrix<Complex>(30, 30);
        ComplexMultiplyContract<ScalarCase>.Fill(a, seed: 31);
        ComplexMultiplyContract<ScalarCase>.Fill(b, seed: 32);
        ComplexMultiplyContract<ScalarCase>.Fill(c, seed: 33);
        var before = c.Clone();

        using var workspace = new Workspace(multithreaded: false);
        TensileLimits.MaxElements = 500;

        Assert.Throws<AllocationLimitException>(
            () => workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View, Complex.One, Complex.One));

        for (int j = 0; j < 30; j++)
            for (int i = 0; i < 30; i++)
                Assert.Equal(before[i, j], c[i, j]);
    }

    [Fact]
    public void RetainedBuffersNeedNoFurtherAllocation()
    {
        var a = new Matrix<Complex>(30, 30);
        var b = new Matrix<Complex>(30, 30);
        var c = new Matrix<Complex>(30, 30);

        using var workspace = new Workspace(multithreaded: false);
        workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View);

        TensileLimits.MaxElements = 500;
        workspace.Multiply(a.ReadOnlyView, b.ReadOnlyView, c.View);
    }
}
