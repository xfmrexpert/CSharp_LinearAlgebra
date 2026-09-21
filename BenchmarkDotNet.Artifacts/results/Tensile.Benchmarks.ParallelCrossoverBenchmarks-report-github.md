```

BenchmarkDotNet v0.15.8, Linux Ubuntu 24.04.4 LTS (Noble Numbat)
Intel Xeon Processor 2.80GHz, 1 CPU, 4 logical and 4 physical cores
.NET SDK 10.0.112
  [Host] : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4
  Dry    : .NET 10.0.12 (10.0.12, 10.0.1226.42308), X64 RyuJIT x86-64-v4

Job=Dry  IterationCount=1  LaunchCount=1  
RunStrategy=ColdStart  UnrollFactor=1  WarmupCount=1  

```
| Method   | Shape   | Mean      | Error | Ratio |
|--------- |-------- |----------:|------:|------:|
| **Serial**   | **1024x64** |  **9.155 ms** |    **NA** |  **1.00** |
| Threaded | 1024x64 | 19.133 ms |    NA |  2.09 |
|          |         |           |       |       |
| **Serial**   | **128**     |  **6.359 ms** |    **NA** |  **1.00** |
| Threaded | 128     | 13.508 ms |    NA |  2.12 |
|          |         |           |       |       |
| **Serial**   | **160**     |  **9.143 ms** |    **NA** |  **1.00** |
| Threaded | 160     | 13.875 ms |    NA |  1.52 |
|          |         |           |       |       |
| **Serial**   | **192**     |  **6.212 ms** |    **NA** |  **1.00** |
| Threaded | 192     | 14.664 ms |    NA |  2.36 |
|          |         |           |       |       |
| **Serial**   | **2048x64** | **20.540 ms** |    **NA** |  **1.00** |
| Threaded | 2048x64 | 25.852 ms |    NA |  1.26 |
|          |         |           |       |       |
| **Serial**   | **256**     |  **6.759 ms** |    **NA** |  **1.00** |
| Threaded | 256     | 17.080 ms |    NA |  2.53 |
|          |         |           |       |       |
| **Serial**   | **256x64**  |  **5.769 ms** |    **NA** |  **1.00** |
| Threaded | 256x64  | 13.473 ms |    NA |  2.34 |
|          |         |           |       |       |
| **Serial**   | **384**     |  **9.317 ms** |    **NA** |  **1.00** |
| Threaded | 384     | 20.297 ms |    NA |  2.18 |
|          |         |           |       |       |
| **Serial**   | **512x64**  |  **6.609 ms** |    **NA** |  **1.00** |
| Threaded | 512x64  | 17.252 ms |    NA |  2.61 |
|          |         |           |       |       |
| **Serial**   | **64**      |  **4.372 ms** |    **NA** |  **1.00** |
| Threaded | 64      | 12.188 ms |    NA |  2.79 |
|          |         |           |       |       |
| **Serial**   | **96**      |  **5.083 ms** |    **NA** |  **1.00** |
| Threaded | 96      | 12.907 ms |    NA |  2.54 |
