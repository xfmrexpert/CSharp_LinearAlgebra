```

BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
Intel Xeon Processor 2.80GHz, 1 CPU, 4 logical and 4 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4
  Dry    : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4

Job=Dry  IterationCount=1  LaunchCount=1  
RunStrategy=ColdStart  UnrollFactor=1  WarmupCount=1  

```
| Method                | N    | Mean       | Error | Ratio | Allocated | Alloc Ratio |
|---------------------- |----- |-----------:|------:|------:|----------:|------------:|
| **&#39;native pointers&#39;**     | **128**  |   **5.767 ms** |    **NA** |  **1.00** |         **-** |          **NA** |
| &#39;POH storage, direct&#39; | 128  |   6.162 ms |    NA |  1.07 |         - |          NA |
| &#39;public API&#39;          | 128  |   7.874 ms |    NA |  1.37 |         - |          NA |
|                       |      |            |       |       |           |             |
| **&#39;native pointers&#39;**     | **512**  |  **23.520 ms** |    **NA** |  **1.00** |   **28768 B** |        **1.00** |
| &#39;POH storage, direct&#39; | 512  |  22.710 ms |    NA |  0.97 |   26800 B |        0.93 |
| &#39;public API&#39;          | 512  |  25.231 ms |    NA |  1.07 |   26424 B |        0.92 |
|                       |      |            |       |       |           |             |
| **&#39;native pointers&#39;**     | **2048** | **210.520 ms** |    **NA** |  **1.00** |  **286968 B** |        **1.00** |
| &#39;POH storage, direct&#39; | 2048 | 200.392 ms |    NA |  0.95 |  283984 B |        0.99 |
| &#39;public API&#39;          | 2048 | 176.802 ms |    NA |  0.84 |  285312 B |        0.99 |
