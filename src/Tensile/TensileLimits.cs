namespace Tensile;

/// <summary>
/// Process-wide resource policy.
///
/// Every allocation the library makes on a caller's behalf -- a matrix's
/// storage, a result, a work panel -- passes through one internal path that
/// compares the request against <see cref="MaxElements"/> before asking the
/// runtime for memory. A request over the limit is refused at the door with an
/// <see cref="AllocationLimitException"/> naming both numbers, rather than
/// attempted and left to fail wherever the runtime notices.
///
/// This is a policy control, not a memory-safety one. An over-large request
/// already fails cleanly; what the limit adds is that a service can decide
/// that a 16 GB matrix is a refusal and not an attempt, and can tell the two
/// apart in its error handling. The default is the largest single allocation
/// the runtime permits, which is to say no policy at all: a library should not
/// guess a service's memory budget.
///
/// The limit counts elements, not bytes, and applies to each allocation
/// separately. A <c>Matrix&lt;double&gt;</c> of n elements and a
/// <c>Matrix&lt;Half&gt;</c> of n elements are the same request to it, and an
/// operation that allocates several buffers is bounded per buffer, not in
/// total. Set it once, at startup; it is a static, and changing it while work
/// is in flight affects whichever allocation comes next.
/// </summary>
public static class TensileLimits
{
    private static int _maxElements = Array.MaxLength;

    /// <summary>
    /// The most elements a single allocation may request. Defaults to
    /// <see cref="Array.MaxLength"/>, the runtime's own ceiling, and can be
    /// set lower but never higher.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">The value is not positive or exceeds <see cref="Array.MaxLength"/>.</exception>
    public static int MaxElements
    {
        get => Volatile.Read(ref _maxElements);
        set
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
            ArgumentOutOfRangeException.ThrowIfGreaterThan(value, Array.MaxLength);

            Volatile.Write(ref _maxElements, value);
        }
    }

    /// <summary>Restore the default: no policy limit below the runtime's own.</summary>
    public static void Reset() => Volatile.Write(ref _maxElements, Array.MaxLength);

    /// <summary>
    /// Refuse a request over the limit. Called by the allocator for every
    /// allocation, and available to code that sizes a buffer some other way.
    /// </summary>
    /// <param name="elements">Elements about to be requested.</param>
    /// <param name="purpose">What the allocation is for, for the message.</param>
    /// <exception cref="AllocationLimitException">The request exceeds <see cref="MaxElements"/>.</exception>
    internal static void Check(long elements, string purpose)
    {
        int limit = MaxElements;

        if (elements > limit)
            throw new AllocationLimitException(elements, limit, purpose);
    }
}

/// <summary>
/// A request for more elements than <see cref="TensileLimits.MaxElements"/>
/// allows. Thrown before any memory is requested, so nothing was allocated and
/// nothing was written.
///
/// Distinct from <see cref="OutOfMemoryException"/> on purpose: that means the
/// runtime tried and failed; this means the library declined to try, because
/// the process said so. A service can map this to "request too large" and the
/// other to "we are in trouble".
/// </summary>
public sealed class AllocationLimitException : Exception
{
    /// <summary>Elements the operation asked for.</summary>
    public long Requested { get; }

    /// <summary>The limit in force when it asked.</summary>
    public int Limit { get; }

    /// <summary>Create the exception for a refused request.</summary>
    /// <param name="requested">Elements the operation asked for.</param>
    /// <param name="limit">The limit in force.</param>
    /// <param name="purpose">What the allocation was for.</param>
    public AllocationLimitException(long requested, int limit, string purpose)
        : base($"Refused to allocate {requested} elements for {purpose}: TensileLimits.MaxElements is {limit}.")
    {
        Requested = requested;
        Limit = limit;
    }
}

/// <summary>
/// The one allocation path. Nothing else in the public assembly writes
/// <c>new T[]</c> or calls <see cref="GC.AllocateArray{T}"/> for storage sized
/// by a request; everything comes through here so that the policy in
/// <see cref="TensileLimits"/> is applied exactly once per allocation and
/// cannot be forgotten at a new call site.
///
/// The kernel assembly allocates too, but only work bounded by the dimensions
/// of operands the caller already holds -- a pivot array of min(m, n), a row
/// sum of m, packing buffers of a few block constants times m -- never by a
/// number the caller has not already paid for. Those are outside the policy by
/// design, and this comment is where that boundary is written down.
/// </summary>
internal static class Storage
{
    /// <summary>
    /// A zeroed array on the pinned object heap, for storage a kernel will be
    /// handed. Pinned so its address never changes, which is what lets a
    /// matrix align its first element once and keep the alignment.
    ///
    /// The policy is measured against <paramref name="elements"/>, the
    /// caller's request; <paramref name="padding"/> is the library's own cost
    /// of alignment and is allocated on top without counting. A limit set to
    /// n therefore admits a matrix of exactly n.
    /// </summary>
    /// <param name="elements">Elements requested.</param>
    /// <param name="padding">Extra elements for alignment. The caller has checked that the sum fits <see cref="Array.MaxLength"/>.</param>
    /// <param name="purpose">What the allocation is for, for the refusal message.</param>
    /// <exception cref="AllocationLimitException">The request exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    public static T[] Pinned<T>(int elements, int padding, string purpose) where T : unmanaged
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elements);
        ArgumentOutOfRangeException.ThrowIfNegative(padding);
        TensileLimits.Check(elements, purpose);

        return GC.AllocateArray<T>(elements + padding, pinned: true);
    }

    /// <summary>A zeroed managed array, for work the library keeps to itself.</summary>
    /// <param name="elements">Elements to allocate.</param>
    /// <param name="purpose">What the allocation is for, for the refusal message.</param>
    /// <exception cref="AllocationLimitException">The request exceeds <see cref="TensileLimits.MaxElements"/>.</exception>
    public static T[] Array<T>(int elements, string purpose)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elements);
        TensileLimits.Check(elements, purpose);

        return new T[elements];
    }
}
