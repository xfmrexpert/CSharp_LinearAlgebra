using System.Runtime.InteropServices;

namespace GemmLab;

public sealed unsafe class Blis : IDisposable
{
    private nint _library;
    private readonly nint _gemm;

    public string LibraryName { get; }
    public string Version { get; }
    public string Architecture { get; }
    public string GemmKernelImplementation { get; }
    public int IntegerBits { get; }
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

    public static Blis? TryLoad(out string reason)
    {
        string? libraryOverride = Environment.GetEnvironmentVariable("GEMMLAB_BLIS_LIBRARY");
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

        reason = $"Could not load {string.Join(" or ", candidates)}. Install BLIS or set GEMMLAB_BLIS_LIBRARY to its shared library path.";
        return null;
    }

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

    public void Dispose()
    {
        if (_library == 0) return;
        NativeLibrary.Free(_library);
        _library = 0;
    }
}