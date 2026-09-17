using System.Runtime.InteropServices;

namespace Tensile.Interop;

/// <summary>
/// Optional binding to a native BLIS <c>bli_dgemm</c>, used as a performance
/// baseline rather than as a backend: nothing in the library depends on it, and
/// it is skipped with a diagnostic when the shared library is absent.
///
/// BLIS is BSD-3 and so passes the forkability test this project applies to
/// dependencies. The dispatch and ABI properties exist because a comparison
/// against a <c>generic</c> BLIS build measures a fallback and is worthless:
/// check <see cref="Architecture"/> and <see cref="GemmKernelImplementation"/>
/// before quoting any ratio.
///
/// Parked in the kernel assembly, internal, because the public assembly no
/// longer compiles unsafe code and this is the one path in the library that
/// loads native code from a path read out of the environment. Phase 4 of
/// docs/security-design.md moves it to its own opt-in package; until then the
/// benchmarks and diagnostics reach it through InternalsVisibleTo and a
/// consumer of the Tensile package cannot reach it at all.
/// </summary>
internal sealed unsafe class Blis : IDisposable
{
    private nint _library;
    private readonly nint _gemm;

    /// <summary>File name of the shared library that was actually loaded.</summary>
    public string LibraryName { get; }

    /// <summary>BLIS version string, as reported by the library.</summary>
    public string Version { get; }

    /// <summary>
    /// Sub-configuration BLIS selected for this CPU, for example <c>haswell</c>.
    /// A value of <c>generic</c> means no optimized kernel was chosen.
    /// </summary>
    public string Architecture { get; }

    /// <summary>
    /// Whether the double GEMM micro-kernel is <c>optimized</c> or a reference
    /// fallback. Anything but <c>optimized</c> invalidates a comparison.
    /// </summary>
    public string GemmKernelImplementation { get; }

    /// <summary>Width of the integer type in the BLIS ABI, 32 or 64.</summary>
    public int IntegerBits { get; }

    /// <summary>Thread count BLIS is configured to use.</summary>
    public long Threads { get; }

    private Blis(nint library, string libraryName)
    {
        _library = library;
        LibraryName = libraryName;
        _gemm = NativeLibrary.GetExport(library, "bli_dgemm");

        var getIntegerBits = (delegate* unmanaged[Cdecl]<byte*>)NativeLibrary.GetExport(
            library, "bli_info_get_int_type_size_str");
        IntegerBits = Marshal.PtrToStringAnsi((nint)getIntegerBits()) switch
        {
            "32" => 32,
            "64" => 64,
            var value => throw new NotSupportedException($"Unsupported BLIS integer ABI: {value}.")
        };

        var getVersion = (delegate* unmanaged[Cdecl]<byte*>)NativeLibrary.GetExport(
            library, "bli_info_get_version_str");
        Version = Marshal.PtrToStringAnsi((nint)getVersion()) ?? "unknown";

        nint setThreads = NativeLibrary.GetExport(library, "bli_thread_set_num_threads");
        nint getThreads = NativeLibrary.GetExport(library, "bli_thread_get_num_threads");
        if (IntegerBits == 64)
        {
            ((delegate* unmanaged[Cdecl]<long, void>)setThreads)(1);
            Threads = ((delegate* unmanaged[Cdecl]<long>)getThreads)();
        }
        else
        {
            ((delegate* unmanaged[Cdecl]<int, void>)setThreads)(1);
            Threads = ((delegate* unmanaged[Cdecl]<int>)getThreads)();
        }

        if (Threads != 1)
            throw new NotSupportedException($"BLIS did not accept single-thread mode (reported {Threads}).");

        var queryArchitecture = (delegate* unmanaged[Cdecl]<int>)NativeLibrary.GetExport(
            library, "bli_arch_query_id");
        var architectureString = (delegate* unmanaged[Cdecl]<int, byte*>)NativeLibrary.GetExport(
            library, "bli_arch_string");
        Architecture = Marshal.PtrToStringUTF8((nint)architectureString(queryArchitecture())) ?? "unknown";

        const int BlisNativeMethod = 1;
        const int BlisDouble = 2;
        var kernelImplementation = (delegate* unmanaged[Cdecl]<int, int, byte*>)NativeLibrary.GetExport(
            library, "bli_info_get_gemm_ukr_impl_string");
        GemmKernelImplementation = Marshal.PtrToStringUTF8(
            (nint)kernelImplementation(BlisNativeMethod, BlisDouble)) switch
        {
            "optimzd" => "optimized",
            { } implementation => implementation,
            null => "unknown"
        };
    }

    /// <summary>
    /// Attempt to load BLIS from the usual platform names, or from the path in
    /// the <c>TENSILE_BLIS_LIBRARY</c> environment variable.
    /// </summary>
    /// <param name="reason">Why the load failed, when the result is null.</param>
    /// <returns>The loaded library, or null if it could not be found.</returns>
    public static Blis? TryLoad(out string reason)
    {
        string? libraryOverride = Environment.GetEnvironmentVariable("TENSILE_BLIS_LIBRARY");
        string[] candidates = !string.IsNullOrWhiteSpace(libraryOverride)
            ? new[] { libraryOverride }
            : OperatingSystem.IsWindows() ? new[] { "blis.dll", "libblis.dll" }
            : OperatingSystem.IsMacOS() ? new[] { "libblis.dylib" }
            : new[] { "libblis.so", "libblis.so.4" };

        foreach (string candidate in candidates)
        {
            if (!NativeLibrary.TryLoad(candidate, out nint library)) continue;

            try
            {
                var backend = new Blis(library, candidate);
                reason = "";
                return backend;
            }
            catch (Exception error) when (error is EntryPointNotFoundException or NotSupportedException)
            {
                NativeLibrary.Free(library);
                reason = $"{candidate}: {error.Message}";
                return null;
            }
        }

        reason = $"Could not load {string.Join(" or ", candidates)}. Install BLIS or set TENSILE_BLIS_LIBRARY to its shared library path.";
        return null;
    }

    /// <summary>
    /// C := beta*C + alpha*A*B through native BLIS, column-major with unit row
    /// stride, matching the signature of the managed drivers.
    /// </summary>
    /// <exception cref="ObjectDisposedException">The library has been unloaded.</exception>
    public void Multiply(
        int rows, int columns, int depth,
        double alpha, double* left, int leftStride,
        double* right, int rightStride,
        double beta, double* result, int resultStride)
    {
        ObjectDisposedException.ThrowIf(_library == 0, this);

        if (IntegerBits == 64)
        {
            var multiply = (delegate* unmanaged[Cdecl]<
                int, int, long, long, long,
                double*, double*, long, long, double*, long, long,
                double*, double*, long, long, void>)_gemm;
            multiply(0, 0, rows, columns, depth, &alpha,
                left, 1, leftStride, right, 1, rightStride, &beta, result, 1, resultStride);
        }
        else
        {
            var multiply = (delegate* unmanaged[Cdecl]<
                int, int, int, int, int,
                double*, double*, int, int, double*, int, int,
                double*, double*, int, int, void>)_gemm;
            multiply(0, 0, rows, columns, depth, &alpha,
                left, 1, leftStride, right, 1, rightStride, &beta, result, 1, resultStride);
        }
    }

    /// <summary>Unload the shared library. Safe to call more than once.</summary>
    public void Dispose()
    {
        if (_library == 0) return;
        NativeLibrary.Free(_library);
        _library = 0;
    }
}