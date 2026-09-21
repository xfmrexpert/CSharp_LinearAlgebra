```

BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
Intel Xeon Processor 2.80GHz, 1 CPU, 4 logical and 4 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4
  Dry    : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4

Job=Dry  IterationCount=1  LaunchCount=1  
RunStrategy=ColdStart  UnrollFactor=1  WarmupCount=1  

```
| Method | Mc  | Kc  | N    | Mean       | Error |
|------- |---- |---- |----- |-----------:|------:|
| **Serial** | **144** | **256** | **128**  |   **5.596 ms** |    **NA** |
| **Serial** | **144** | **256** | **512**  |  **12.956 ms** |    **NA** |
| **Serial** | **144** | **256** | **2048** | **356.094 ms** |    **NA** |
| **Serial** | **144** | **384** | **128**  |   **5.638 ms** |    **NA** |
| **Serial** | **144** | **384** | **512**  |  **13.989 ms** |    **NA** |
| **Serial** | **144** | **384** | **2048** | **419.814 ms** |    **NA** |
| **Serial** | **288** | **256** | **128**  |   **5.662 ms** |    **NA** |
| **Serial** | **288** | **256** | **512**  |  **12.344 ms** |    **NA** |
| **Serial** | **288** | **256** | **2048** | **342.951 ms** |    **NA** |
| **Serial** | **288** | **384** | **128**  |   **5.650 ms** |    **NA** |
| **Serial** | **288** | **384** | **512**  |  **13.364 ms** |    **NA** |
| **Serial** | **288** | **384** | **2048** | **414.077 ms** |    **NA** |
