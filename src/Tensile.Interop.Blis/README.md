# Tensile.Interop.Blis

An optional binding to a native [BLIS](https://github.com/flame/blis)
`bli_dgemm`, used by Tensile's benchmarks as the baseline the managed GEMM is
measured against. Nothing in Tensile depends on it. It exists so that a ratio
against BLIS can be quoted with the architecture BLIS actually dispatched to.

## What it does that the core does not

This package loads and executes native code at run time. `Blis.TryLoad` looks
for `libblis.so` / `libblis.dylib` / `blis.dll` on the usual search path, or —
if the `TENSILE_BLIS_LIBRARY` environment variable is set — loads **exactly
that path** and nothing else.

That environment variable is a benchmarking convenience for development
machines. It means a process that calls `TryLoad` will `dlopen` and run
whatever library the environment names. Do not ship it in a service, a
plugin host, or anything whose environment an untrusted party can influence.
The core `Tensile` package carries no such path: that is invariant I8 of its
security design, and this package is separate precisely so a consumer who
never asked for a native comparison never has one.

## Reading a comparison

Check `Blis.Architecture` and `Blis.GemmKernelImplementation` before quoting
any number. A `generic` architecture or a non-`optimized` kernel means the
BLIS build is a reference fallback and the comparison is against nothing.
`tensile-diag` prints both.

Licence: BSD-3-Clause, the same as Tensile and as BLIS itself.
