```

BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
Intel Xeon Processor 2.80GHz, 1 CPU, 4 logical and 4 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4
  Dry    : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4

Job=Dry  IterationCount=1  LaunchCount=1  
RunStrategy=ColdStart  UnrollFactor=1  WarmupCount=1  

```
| Method   | N    | Threads | Mean      | Error |
|--------- |----- |-------- |----------:|------:|
| **Threaded** | **2048** | **4**       | **202.30 ms** |    **NA** |
| **Threaded** | **2048** | **2**       | **222.09 ms** |    **NA** |
| **Threaded** | **2048** | **1**       | **398.63 ms** |    **NA** |
| **Threaded** | **512**  | **4**       |  **23.35 ms** |    **NA** |
| **Threaded** | **512**  | **2**       |  **23.37 ms** |    **NA** |
| **Threaded** | **512**  | **1**       |  **23.28 ms** |    **NA** |
