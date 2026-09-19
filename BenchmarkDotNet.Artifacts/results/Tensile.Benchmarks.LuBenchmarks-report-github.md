```

BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
Intel Xeon Processor 2.80GHz, 1 CPU, 4 logical and 4 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4
  Dry    : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4

Job=Dry  IterationCount=1  LaunchCount=1  
RunStrategy=ColdStart  UnrollFactor=1  WarmupCount=1  

```
| Method | N    | BlockSize | Mean      | Error | Allocated  |
|------- |----- |---------- |----------:|------:|-----------:|
| **Factor** | **256**  | **32**        |  **17.25 ms** |    **NA** |    **1.12 KB** |
| **Factor** | **256**  | **64**        |  **17.57 ms** |    **NA** |    **1.12 KB** |
| **Factor** | **256**  | **128**       |  **20.03 ms** |    **NA** |    **1.12 KB** |
| **Factor** | **512**  | **32**        |  **34.18 ms** |    **NA** |   **35.02 KB** |
| **Factor** | **512**  | **64**        |  **37.85 ms** |    **NA** |   **32.09 KB** |
| **Factor** | **512**  | **128**       |  **39.23 ms** |    **NA** |   **16.73 KB** |
| **Factor** | **1024** | **32**        |  **71.23 ms** |    **NA** |  **240.67 KB** |
| **Factor** | **1024** | **64**        |  **69.42 ms** |    **NA** |  **151.11 KB** |
| **Factor** | **1024** | **128**       |  **68.82 ms** |    **NA** |   **80.23 KB** |
| **Factor** | **2048** | **32**        | **195.80 ms** |    **NA** | **1100.16 KB** |
| **Factor** | **2048** | **64**        | **180.32 ms** |    **NA** |  **578.91 KB** |
| **Factor** | **2048** | **128**       | **202.31 ms** |    **NA** |  **290.46 KB** |
