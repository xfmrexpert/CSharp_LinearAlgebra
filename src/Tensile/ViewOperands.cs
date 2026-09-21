using Tensile.Kernels;

namespace Tensile;

/// <summary>
/// Hands a view to the kernel seam. A view and a kernel operand carry the same
/// four facts -- buffer, rows, columns, stride -- and differ only in which
/// assembly defines them, so this is a repackaging and nothing more. It is the
/// only place the public assembly touches the kernel assembly's operand types,
/// and it contains no arithmetic.
/// </summary>
internal static class ViewOperands
{
    /// <summary>A read-only kernel operand over the view's storage.</summary>
    public static Operand ToOperand(this ReadOnlyMatrixView<double> view) =>
        new(view.Buffer, view.Rows, view.Columns, view.Stride);

    /// <summary>A read-only kernel operand over the view's storage.</summary>
    public static Operand ToOperand(this MatrixView<double> view) =>
        new(view.Buffer, view.Rows, view.Columns, view.Stride);

    /// <summary>A writable kernel target over the view's storage.</summary>
    public static Target ToTarget(this MatrixView<double> view) =>
        new(view.Buffer, view.Rows, view.Columns, view.Stride);
}
