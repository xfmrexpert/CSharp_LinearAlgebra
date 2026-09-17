using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Tensile.Kernels;

/// <summary>
/// Cache-line alignment for pinned storage.
///
/// Arrays on the pinned object heap are not guaranteed to start on a 64-byte
/// boundary, so <c>Matrix&lt;T&gt;</c> over-allocates by one cache line and
/// begins its view at the first aligned element. The kernels use unaligned load
/// instructions and would be correct without this; it is kept because the
/// measured results were taken with aligned operands, and changing two things
/// at once would confound the re-measurement.
///
/// This is the only address read on the allocation path, and it lives here
/// rather than beside <c>Matrix&lt;T&gt;</c> because the public assembly
/// compiles with <c>AllowUnsafeBlocks</c> off. Reading the address is sound
/// only because the array is pinned: a movable array's address is stale the
/// moment the garbage collector runs.
/// </summary>
internal static unsafe class Alignment
{
    /// <summary>Alignment target in bytes: one cache line on every x86-64 and ARM64 part that matters.</summary>
    public const int CacheLine = 64;

    /// <summary>
    /// Elements to over-allocate so that some element in the first cache line
    /// is aligned. Ceiling of <c>64 / sizeof(T)</c>; for a type whose size does
    /// not divide 64, exact alignment may not be reachable and the offset lands
    /// as close as element granularity allows.
    /// </summary>
    public static int PaddingElements<T>() where T : unmanaged =>
        (CacheLine + sizeof(T) - 1) / sizeof(T);

    /// <summary>
    /// Index of the first element of <paramref name="pinned"/> on a cache-line
    /// boundary. Never more than <see cref="PaddingElements{T}"/>.
    /// </summary>
    /// <param name="pinned">An array allocated with <c>pinned: true</c>.</param>
    public static int AlignedOffset<T>(T[] pinned) where T : unmanaged
    {
        if (pinned.Length == 0) return 0;

        nuint address = (nuint)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(pinned));
        nuint misalignment = address % CacheLine;

        if (misalignment == 0) return 0;

        nuint bytesToBoundary = CacheLine - misalignment;
        return (int)((bytesToBoundary + (nuint)sizeof(T) - 1) / (nuint)sizeof(T));
    }
}
