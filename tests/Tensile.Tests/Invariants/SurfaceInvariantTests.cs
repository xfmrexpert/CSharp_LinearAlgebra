using System.Reflection;
using System.Runtime.InteropServices;

namespace Tensile.Tests.Invariants;

/// <summary>
/// Invariants about the shape of the public surface, checked by reflection over
/// the shipped assembly rather than by reading the source.
///
/// These are the ones that can regress silently: nothing about adding a
/// pointer-typed public method fails a build or a behavioural test, and the
/// review that would catch it is the same review that missed the last two
/// bugs. Putting the guarantee in the unit suite means the surface cannot
/// drift without a red test saying so.
///
/// Pins I3 (no public pointers), I8 (no native loading in the core), the
/// visibility half of I4 (primitives internal), and the design decision that
/// implements I6 (no disposal to race against).
/// </summary>
public class SurfaceInvariantTests
{
    private static readonly Assembly Core = typeof(Matrix).Assembly;

    /// <summary>
    /// I3: no public type or member accepts, stores or returns a raw pointer.
    /// A caller who has one binds it into a Span themselves, stating its
    /// length, which is the fact the library needs and today never receives.
    /// </summary>
    [Fact]
    public void NoPublicMemberExposesAPointer()
    {
        var offenders = new List<string>();

        foreach (Type type in Core.GetExportedTypes())
        {
            const BindingFlags Declared =
                BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

            foreach (MemberInfo member in type.GetMembers(Declared))
            {
                foreach (Type involved in TypesInvolvedIn(member))
                {
                    if (IsPointerLike(involved))
                    {
                        offenders.Add($"{type.FullName}.{member.Name} ({involved})");
                        break;
                    }
                }
            }
        }

        Assert.True(
            offenders.Count == 0,
            "[I3] public members with pointer types (closed by Phase 3, when the primitives go internal "
            + "and views bind spans):\n  " + string.Join("\n  ", offenders));
    }

    /// <summary>
    /// I3, stated separately because <c>ILinearOperator</c> is a public
    /// extension point -- it is how expm and any matrix-free operator plug in
    /// -- and a pointer in its signature is a pointer every implementer must
    /// handle.
    /// </summary>
    [Fact]
    public void LinearOperatorContractHasNoPointers()
    {
        Type? contract = Core.GetType("Tensile.Primitives.ILinearOperator") ?? Core.GetType("Tensile.ILinearOperator");

        Assert.True(contract is not null, "ILinearOperator should exist somewhere in the core assembly.");

        var offenders = contract!.GetMethods()
            .Where(m => TypesInvolvedIn(m).Any(IsPointerLike))
            .Select(m => m.Name)
            .ToList();

        Assert.True(
            offenders.Count == 0,
            "[I3] ILinearOperator members taking pointers (closed by Phase 3, re-signatured over views): "
            + string.Join(", ", offenders));
    }

    /// <summary>
    /// I4, visibility half: the unvalidated layer is not part of the public
    /// API. The compiler half (no unsafe code in this assembly at all) cannot
    /// be observed by reflection and is held by AllowUnsafeBlocks=false.
    /// </summary>
    [Fact]
    public void PrimitivesNamespaceHasNoPublicTypes()
    {
        var exposed = Core.GetExportedTypes()
            .Where(t => t.Namespace == "Tensile.Primitives")
            .Select(t => t.Name)
            .OrderBy(n => n)
            .ToList();

        Assert.True(
            exposed.Count == 0,
            "[I4] public types in Tensile.Primitives (closed by Phase 3, when they move to Tensile.Kernels as internal): "
            + string.Join(", ", exposed));
    }

    /// <summary>
    /// I8: the core assembly loads no native code. The BLIS binding, which
    /// dlopens a path read from an environment variable, ships in a separate
    /// opt-in package, so a consumer who never asked for it never carries it.
    /// </summary>
    [Fact]
    public void InteropIsNotInTheCoreAssembly()
    {
        var present = Core.GetTypes()
            .Where(t => t.Namespace is not null && t.Namespace.StartsWith("Tensile.Interop", StringComparison.Ordinal))
            .Select(t => t.FullName)
            .ToList();

        Assert.True(
            present.Count == 0,
            "[I8] interop types in the core assembly (closed by Phase 4, separate Tensile.Interop.Blis package): "
            + string.Join(", ", present));
    }

    /// <summary>
    /// I8, the other route: no P/Invoke declarations. Green today -- BLIS uses
    /// NativeLibrary and function pointers rather than DllImport -- and kept as
    /// a guard so the route cannot be reopened.
    /// </summary>
    [Fact]
    public void CoreDeclaresNoPInvoke()
    {
        const BindingFlags All =
            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static | BindingFlags.DeclaredOnly;

        var imports = Core.GetTypes()
            .SelectMany(t => t.GetMethods(All))
            .Where(m => m.GetCustomAttribute<DllImportAttribute>() is not null)
            .Select(m => $"{m.DeclaringType?.FullName}.{m.Name}")
            .ToList();

        Assert.True(imports.Count == 0, "[I8] DllImport in the core assembly: " + string.Join(", ", imports));
    }

    /// <summary>
    /// I6 by construction: with GC-tracked pinned storage there is no Dispose,
    /// so there is nothing to race a live view against. This test pins the
    /// design decision that makes the invariant hold without a caller-side
    /// rule; the invariant itself has no failure mode to test once this passes.
    /// </summary>
    [Fact]
    public void MatrixStorageIsNotDisposable()
    {
        Assert.False(
            typeof(IDisposable).IsAssignableFrom(typeof(Matrix<>)),
            "[I6] Matrix<T> implements IDisposable, so a view can outlive its storage (closed by Phase 2, POH storage).");
    }

    /// <summary>Same decision, for the type that owns a matrix on the caller's behalf.</summary>
    [Fact]
    public void FactorizationIsNotDisposable()
    {
        Assert.False(
            typeof(IDisposable).IsAssignableFrom(typeof(LuDecomposition)),
            "[I6] LuDecomposition implements IDisposable (closed by Phase 2, when it owns a Matrix<double> on the POH).");
    }

    private static IEnumerable<Type> TypesInvolvedIn(MemberInfo member)
    {
        switch (member)
        {
            case MethodInfo method:
                yield return method.ReturnType;
                foreach (ParameterInfo p in method.GetParameters()) yield return p.ParameterType;
                break;

            case ConstructorInfo constructor:
                foreach (ParameterInfo p in constructor.GetParameters()) yield return p.ParameterType;
                break;

            case PropertyInfo property:
                yield return property.PropertyType;
                foreach (ParameterInfo p in property.GetIndexParameters()) yield return p.ParameterType;
                break;

            case FieldInfo field:
                yield return field.FieldType;
                break;

            case EventInfo @event when @event.EventHandlerType is not null:
                yield return @event.EventHandlerType;
                break;
        }
    }

    /// <summary>A pointer, a function pointer, or an array/byref of one.</summary>
    private static bool IsPointerLike(Type type)
    {
        for (Type? current = type; current is not null; current = current.HasElementType ? current.GetElementType() : null)
        {
            if (current.IsPointer || current.IsFunctionPointer) return true;
            if (!current.HasElementType) return false;
        }

        return false;
    }
}
