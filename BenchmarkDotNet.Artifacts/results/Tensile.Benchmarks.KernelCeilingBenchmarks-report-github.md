```

BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
Intel Xeon Processor 2.10GHz, 1 CPU, 4 logical and 4 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4
  Dry    : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4

Job=Dry  IterationCount=1  LaunchCount=1  
RunStrategy=ColdStart  UnrollFactor=1  WarmupCount=1  

```
| Method  | Kernel       | Mean     | Error |
|-------- |------------- |---------:|------:|
| **Ceiling** | **AVX-512 16x8** | **21.10 ms** |    **NA** |
| **Ceiling** | **AVX2/FMA 8x6** | **13.50 ms** |    **NA** |
| **Ceiling** | **scalar 4x4**   | **20.99 ms** |    **NA** |
