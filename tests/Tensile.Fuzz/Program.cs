using System.Buffers.Binary;
using SharpFuzz;

namespace Tensile.Fuzz;

/// <summary>
/// The fuzz harness: an input is a script of operations over hostile
/// integers, and the property under test is invariant I5 from
/// docs/security-design.md -- every public entry point either produces its
/// documented result or throws one of its documented exceptions. Anything
/// else escaping (IndexOutOfRange, Overflow, OutOfMemory, NullReference, an
/// AccessViolation that takes the process down) is a finding.
///
/// The property tests in tests/Tensile.Tests/Invariants enumerate the values
/// somebody thought of. This finds the ones nobody did. Its seed corpus is
/// those enumerated cases, so the fuzzer starts from the known edges and
/// mutates outward.
///
/// Three modes, chosen by the first argument:
///
///   (none)          run under afl-fuzz via SharpFuzz's out-of-process loop
///   --replay FILE.. run each file once and report; how CI checks a crash
///                   reproduces, and how a developer debugs one
///   --self-check    run the shipped corpus; fails if any seed is itself a
///                   finding, which is the cheap gate the per-PR build runs
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        // Nothing from the instrumented assembly may run before the fuzzer
        // loop attaches to afl's shared memory: an instrumented method writes
        // coverage into that map, and before Run has mapped it the write is
        // an access violation. Script.Run sets the policy limit itself.

        if (args.Length > 0 && args[0] == "--replay")
        {
            int failures = 0;

            foreach (string path in args.Skip(1))
            {
                Console.Write($"{path}: ");
                failures += Replay(File.ReadAllBytes(path)) ? 0 : 1;
            }

            return failures == 0 ? 0 : 1;
        }

        if (args.Length > 0 && args[0] == "--self-check")
        {
            string corpus = Path.Combine(AppContext.BaseDirectory, "Corpus");
            string[] seeds = Directory.GetFiles(corpus);
            int failures = 0;

            foreach (string seed in seeds)
            {
                Console.Write($"{Path.GetFileName(seed)}: ");
                failures += Replay(File.ReadAllBytes(seed)) ? 0 : 1;
            }

            Console.WriteLine($"{seeds.Length} seeds, {failures} findings");
            return failures == 0 && seeds.Length > 0 ? 0 : 1;
        }

        Fuzzer.OutOfProcess.Run(stream =>
        {
            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);
            Script.Run(buffer.ToArray());
        });

        return 0;
    }

    private static bool Replay(byte[] input)
    {
        try
        {
            Script.Run(input);
            Console.WriteLine("ok");
            return true;
        }
        catch (Exception error)
        {
            Console.WriteLine($"FINDING {error.GetType().Name}: {error.Message}");
            Console.WriteLine(error.StackTrace);
            return false;
        }
    }
}

/// <summary>
/// Decodes an input as a sequence of operations and runs them against the
/// public API, swallowing only the exceptions the API documents.
/// </summary>
internal static class Script
{
    /// <summary>Largest dimension the script will hand a constructor; keeps the run's time bounded.</summary>
    private const int MaxDimension = 96;

    /// <summary>
    /// Ceiling for the run. A script can ask for products and copies of
    /// millions of elements; anything beyond this is refused by policy rather
    /// than attempted, which keeps the fuzzer's memory flat and exercises the
    /// refusal path constantly.
    /// </summary>
    private const int MaxElements = 1 << 20;

    public static void Run(ReadOnlySpan<byte> input)
    {
        TensileLimits.MaxElements = MaxElements;

        var reader = new Reader(input);

        // Two matrices live across the script so operations can combine them.
        Matrix<double> a = Matrix.Identity<double>(4);
        Matrix<double> b = Matrix.Identity<double>(4);

        while (reader.Remaining >= 1)
        {
            byte op = reader.Byte();

            try
            {
                switch (op % 14)
                {
                    case 0:
                        _ = new MatrixShape(reader.Int(), reader.Int(), reader.Int());
                        break;

                    case 1:
                    {
                        // Bind over a caller array whose length is itself hostile.
                        var shape = new MatrixShape(reader.Small(), reader.Small(), reader.Small());
                        var buffer = new double[Math.Clamp(reader.Int(), 0, 8192)];
                        MatrixView<double> view = MatrixView<double>.Bind(buffer, shape);
                        view.Fill(1.0);
                        break;
                    }

                    case 2:
                        a = new Matrix<double>(reader.Dimension(), reader.Dimension(), reader.Small());
                        Fill(a, reader.Byte());
                        break;

                    case 3:
                        b = new Matrix<double>(reader.Dimension(), reader.Dimension(), reader.Small());
                        Fill(b, reader.Byte());
                        break;

                    case 4:
                    {
                        MatrixView<double> block = a.Slice(reader.Int(), reader.Int(), reader.Int(), reader.Int());
                        block.Fill(2.0);
                        break;
                    }

                    case 5:
                        _ = a.ReadOnlyView.Slice(reader.Int(), reader.Int(), reader.Int(), reader.Int()).ToArray();
                        break;

                    case 6:
                        _ = a[reader.Int(), reader.Int()];
                        break;

                    case 7:
                        _ = a.Column(reader.Int());
                        break;

                    case 8:
                    {
                        // Overlapping copy within one matrix: the review finding's class.
                        MatrixView<double> source = a.Slice(reader.Small(), reader.Small(), reader.Small(), reader.Small());
                        MatrixView<double> target = a.Slice(reader.Small(), reader.Small(), source.Rows, source.Columns);
                        source.CopyTo(target);
                        break;
                    }

                    case 9:
                        b = a.Multiply(b);
                        break;

                    case 10:
                    {
                        LuDecomposition lu = a.FactorLu(reader.Int());
                        _ = lu.Solve(b);
                        _ = lu.ReciprocalCondition(reader.Int());
                        _ = lu.Determinant();
                        break;
                    }

                    case 11:
                        _ = a.EstimateOneNorm(reader.Small(), reader.Int());
                        break;

                    case 12:
                        _ = Matrix.FromColumnMajor<double>(reader.Int(), reader.Int(), new double[Math.Clamp(reader.Int(), 0, 64)]);
                        break;

                    case 13:
                    {
                        a.As<UpperTriangular>().SolveInPlace(b.View);
                        a.As<UnitLowerTriangular>().SolveTransposedInPlace(b.View);
                        break;
                    }
                }
            }
            catch (ArgumentException)
            {
                // Documented: a bad argument. Includes ArgumentOutOfRange and ArgumentNull.
            }
            catch (InvalidOperationException)
            {
                // Documented: a singular or non-square factorization.
            }
            catch (AllocationLimitException)
            {
                // Documented: refused by policy.
            }
        }
    }

    private static void Fill(Matrix<double> m, byte pattern)
    {
        var rng = new Random(pattern);

        for (int j = 0; j < m.Columns; j++)
            for (int i = 0; i < m.Rows; i++)
                m[i, j] = rng.NextDouble() - 0.5 + (i == j ? m.Rows : 0.0);
    }

    /// <summary>Little-endian reads with a zero fill past the end, so a short input is still a script.</summary>
    private ref struct Reader(ReadOnlySpan<byte> input)
    {
        private readonly ReadOnlySpan<byte> _input = input;
        private int _at;

        public int Remaining => _input.Length - _at;

        public byte Byte() => Remaining >= 1 ? _input[_at++] : (byte)0;

        /// <summary>A raw 32-bit value: hostile by construction.</summary>
        public int Int()
        {
            if (Remaining < 4) { _at = _input.Length; return 0; }

            int value = BinaryPrimitives.ReadInt32LittleEndian(_input.Slice(_at));
            _at += 4;
            return value;
        }

        /// <summary>A value in [0, 255], for shapes that should usually be valid.</summary>
        public int Small() => Byte();

        /// <summary>A dimension the harness is prepared to allocate.</summary>
        public int Dimension() => Byte() % (MaxDimension + 1);
    }
}
