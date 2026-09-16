using System.Diagnostics;
using System.Runtime.InteropServices;

namespace GemmLab;

/// <summary>
/// Verification and cost reporting for <see cref="NormEstimate"/> and
/// <see cref="Condition"/>.
///
/// The estimator is a lower bound, so "correct" cannot mean "equals the true
/// norm". What is checked is that it never exceeds the true 1-norm (beyond
/// rounding), that it is exact on the large majority of random matrices, and
/// that it never falls below the factor-of-2 range the algorithm is expected to
/// hold. A run that drifts outside those is a regression even though no single
/// estimate is wrong on its own.
/// </summary>
public static unsafe class NormEstimateTest
{
    private const double Epsilon = 2.220446049250313e-16;

    public static bool Run<TKernel>(bool verifyOnly) where TKernel : struct, IMicroKernel
    {
        Console.WriteLine();
        Console.WriteLine($"=== normest1 / condition estimation (GEMM via {TKernel.Name}) ===");

        if (!TKernel.IsSupported)
        {
            Console.WriteLine("  not supported on this CPU, skipping");
            return true;
        }

        bool passed = VerifyAgainstExact();
        passed &= VerifyPowers<TKernel>();
        passed &= VerifyCondition<TKernel>();

        Console.WriteLine(passed ? "  status        : PASS" : "  status        : FAIL");

        if (!passed) return false;
        if (verifyOnly) return true;

        ReportCost();
        return true;
    }

    /// <summary>
    /// The estimate must never exceed the true 1-norm, and should equal it
    /// almost always.
    /// </summary>
    private static bool VerifyAgainstExact()
    {
        bool passed = true;

        // A non-negative matrix is the case where the algorithm is provably
        // sharp, and so the case that catches an implementation error. With
        // A >= 0 the first sign matrix is all ones, so A^T*S is exactly the
        // vector of column sums and the very first sort lands on the true
        // maximiser. Anything below 100% here is a bug, not bad luck.
        passed &= Ensemble("non-negative", nonNegative: true, columns: 2, requiredExact: 1.00);

        // Random signs are the hard end: the columns of a uniform random matrix
        // have nearly equal 1-norms, so many near-ties compete and the sign
        // vectors carry little information about which column wins.
        passed &= Ensemble("signed, t=2", nonNegative: false, columns: 2, requiredExact: 0.30);
        passed &= Ensemble("signed, t=4", nonNegative: false, columns: 4, requiredExact: 0.45);

        return passed;
    }

    /// <summary>
    /// Run one ensemble and check the two properties that must hold
    /// unconditionally (never an overestimate, never worse than a factor of 2)
    /// plus an exactness floor calibrated to that ensemble.
    /// </summary>
    private static bool Ensemble(string label, bool nonNegative, int columns, double requiredExact)
    {
        int[] sizes = { 4, 9, 16, 33, 64, 129, 256 };
        int cases = 0;
        int exact = 0;
        double worstRatio = double.PositiveInfinity;
        bool overestimated = false;

        foreach (int n in sizes)
        {
            for (int trial = 0; trial < 40; trial++)
            {
                double* a = RandomMatrix(
                    n, n, n, seed: n * 1000 + trial, skewed: trial % 3 == 0, nonNegative: nonNegative);

                try
                {
                    double truth = Norms.One(n, n, a, n);
                    double estimate = NormEstimate.OfMatrix(n, a, n, power: 1, columns: columns).Value;

                    cases++;

                    // Tolerance covers summation order only: the estimate is a
                    // maximum over sums of the same n terms as the exact norm.
                    if (estimate > truth * (1.0 + 64.0 * Epsilon)) overestimated = true;
                    if (estimate >= truth * (1.0 - 64.0 * Epsilon)) exact++;

                    worstRatio = Math.Min(worstRatio, estimate / truth);
                }
                finally
                {
                    NativeMemory.AlignedFree(a);
                }
            }
        }

        double exactFraction = (double)exact / cases;

        Console.WriteLine(
            $"  {label,-14}: {exact,4}/{cases} exact ({exactFraction,6:P1}), worst ratio {worstRatio:F4}");

        bool passed = true;

        if (overestimated)
        {
            Console.WriteLine("  FAIL          : estimate exceeded the true 1-norm (it is a lower bound by construction)");
            passed = false;
        }

        if (worstRatio < 0.5)
        {
            Console.WriteLine($"  FAIL          : worst ratio {worstRatio:F4} is below the expected factor-of-2 floor");
            passed = false;
        }

        if (exactFraction < requiredExact)
        {
            Console.WriteLine($"  FAIL          : {exactFraction:P1} exact is below the {requiredExact:P0} floor for this ensemble");
            passed = false;
        }

        return passed;
    }

    /// <summary>
    /// ||A^p||_1 must match the norm of the explicitly formed power. This is
    /// the path Al-Mohy and Higham's expm needs.
    /// </summary>
    private static bool VerifyPowers<TKernel>() where TKernel : struct, IMicroKernel
    {
        using var gemm = GemmDispatch.Serial<TKernel>();

        bool passed = true;
        double worst = 0.0;

        foreach (int n in new[] { 8, 17, 48 })
        foreach (int power in new[] { 2, 3, 5 })
        {
            // Scaled down so that A^5 stays well inside range.
            double* a = RandomMatrix(n, n, n, seed: n * 31 + power, skewed: false, scale: 1.0 / n);
            double* accumulated = Alloc(n * n);
            double* work = Alloc(n * n);

            try
            {
                new Span<double>(a, n * n).CopyTo(new Span<double>(accumulated, n * n));

                for (int step = 1; step < power; step++)
                {
                    gemm.MultiplySerial<TKernel>(
                        n, n, n, 1.0, accumulated, n, a, n, 0.0, work, n);
                    new Span<double>(work, n * n).CopyTo(new Span<double>(accumulated, n * n));
                }

                double truth = Norms.One(n, n, accumulated, n);
                double estimate = NormEstimate.OfMatrix(n, a, n, power).Value;

                double relative = truth == 0.0 ? 0.0 : Math.Abs(estimate - truth) / truth;
                worst = Math.Max(worst, relative);

                if (estimate > truth * (1.0 + 1e-10))
                {
                    Console.WriteLine($"  FAIL          : n={n} power={power} estimate {estimate:E6} exceeds true {truth:E6}");
                    passed = false;
                }
            }
            finally
            {
                NativeMemory.AlignedFree(a);
                NativeMemory.AlignedFree(accumulated);
                NativeMemory.AlignedFree(work);
            }
        }

        Console.WriteLine($"  matrix powers : worst relative gap to exact ||A^p||_1 = {worst:E3}");
        return passed;
    }

    /// <summary>
    /// rcond on matrices whose conditioning is known: the identity is perfectly
    /// conditioned, Hilbert matrices are famously not.
    /// </summary>
    private static bool VerifyCondition<TKernel>() where TKernel : struct, IMicroKernel
    {
        using var gemm = GemmDispatch.Serial<TKernel>();

        bool passed = true;

        // Identity: cond_1 = 1 exactly.
        {
            const int n = 64;
            double* a = Alloc(n * n);
            new Span<double>(a, n * n).Clear();
            for (int i = 0; i < n; i++) a[(nint)i * n + i] = 1.0;

            try
            {
                double norm = Norms.One(n, n, a, n);
                using var lu = Lu.Factor<TKernel>(n, n, a, n, gemm, 16);
                double rcond = Condition.ReciprocalOne(norm, lu);

                Console.WriteLine($"  identity      : rcond = {rcond:F6} (exact 1.000000)");

                if (Math.Abs(rcond - 1.0) > 1e-12)
                {
                    Console.WriteLine("  FAIL          : identity should give rcond = 1");
                    passed = false;
                }
            }
            finally
            {
                NativeMemory.AlignedFree(a);
            }
        }

        // Hilbert: rcond must fall off a cliff with order.
        Console.WriteLine("  hilbert       : order   rcond estimate");

        double previous = double.PositiveInfinity;

        foreach (int n in new[] { 4, 6, 8, 10, 12 })
        {
            double* a = Alloc(n * n);
            for (int j = 0; j < n; j++)
                for (int i = 0; i < n; i++)
                    a[(nint)j * n + i] = 1.0 / (i + j + 1);

            try
            {
                double norm = Norms.One(n, n, a, n);
                using var lu = Lu.Factor<TKernel>(n, n, a, n, gemm, 8);
                double rcond = Condition.ReciprocalOne(norm, lu);

                Console.WriteLine($"                  {n,5}   {rcond:E3}");

                if (rcond > previous)
                {
                    Console.WriteLine("  FAIL          : rcond should decrease with Hilbert order");
                    passed = false;
                }

                previous = rcond;
            }
            finally
            {
                NativeMemory.AlignedFree(a);
            }
        }

        if (previous > 1e-12)
        {
            Console.WriteLine($"  FAIL          : Hilbert(12) rcond {previous:E3} is implausibly healthy");
            passed = false;
        }

        return passed;
    }

    /// <summary>
    /// What the estimate costs relative to the exact norm it replaces. The
    /// exact 1-norm of a dense matrix is O(n^2) and so is each estimator
    /// product, which is why this is only interesting for operators whose
    /// entries are not available -- A^-1 and A^k.
    /// </summary>
    private static void ReportCost()
    {
        Console.WriteLine("  cost          :    n   iterations   products   est/exact");

        foreach (int n in new[] { 64, 256, 512, 1024 })
        {
            double* a = RandomMatrix(n, n, n, seed: n, skewed: true);

            try
            {
                double truth = Norms.One(n, n, a, n);
                var result = NormEstimate.OfMatrix(n, a, n);

                Console.WriteLine(
                    $"                  {n,5}   {result.Iterations,10}   {result.Products,8}   {result.Value / truth,9:F6}");
            }
            finally
            {
                NativeMemory.AlignedFree(a);
            }
        }
    }

    /// <param name="skewed">
    /// Concentrate mass in a few columns. A matrix with one dominant column is
    /// the easy case for the estimator; uniform random entries are the case
    /// where several columns compete and it has to iterate.
    /// </param>
    private static double* RandomMatrix(
        int rows, int columns, int stride, int seed, bool skewed,
        double scale = 1.0, bool nonNegative = false)
    {
        var rng = new Random(seed);
        double* a = Alloc(stride * columns);

        new Span<double>(a, stride * columns).Clear();

        for (int j = 0; j < columns; j++)
        {
            double weight = skewed && j % 7 == 0 ? 10.0 : 1.0;

            for (int i = 0; i < rows; i++)
            {
                double value = nonNegative ? rng.NextDouble() : rng.NextDouble() - 0.5;
                a[(nint)j * stride + i] = scale * weight * value;
            }
        }

        return a;
    }

    private static double* Alloc(int count) =>
        (double*)NativeMemory.AlignedAlloc((nuint)count * sizeof(double), 64);
}
