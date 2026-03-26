===============================================================================
                    CUDA Warp Architecture Explorer
              GPU Execution Model Benchmarking on NVIDIA A100
===============================================================================

Project Name Suggestion: cuda-warp-architecture-explorer

===============================================================================
TABLE OF CONTENTS
===============================================================================

1. Overview
2. System Architecture
3. Benchmarks Included
4. Results Summary
5. Technical Deep Dive
6. ASPIRE2A HPC Deployment
7. Quick Start Guide
8. Repository Structure
9. Key Learnings
10. License

===============================================================================
1. OVERVIEW
===============================================================================

This project explores low-level GPU execution behavior on NVIDIA A100 GPUs 
through custom CUDA C++ micro-benchmarks. It measures three critical 
performance factors that affect all GPU workloads:

    - Warp Divergence: Performance penalty when threads take different paths
    - Memory Bank Conflicts: Latency spikes from shared memory access patterns
    - Kernel Occupancy: Throughput vs. block size trade-offs

Unlike high-level ML frameworks, this project operates at the kernel level,
providing insights into how the GPU silicon actually executes your code.

===============================================================================
2. SYSTEM ARCHITECTURE
===============================================================================

    +------------------+     +-------------------+     +------------------+
    |  PBS Job Script  | --> |  CUDA Kernels     | --> |  Results (CSV)   |
    |  (Resource Mgmt) |     |  (C++/CUDA)       |     |  + Plots (PNG)   |
    +------------------+     +-------------------+     +------------------+
           |                        |                         |
           v                        v                         v
    ASPIRE2A Scheduler      NVIDIA A100 GPU          Python Analysis
    (1 GPU:16 CPU:110GB)    (Ampere Architecture)    (Matplotlib/Pandas)

Hardware Target:
    - GPU: NVIDIA A100-SXM4-40GB (Ampere Architecture)
    - CUDA Version: 12.4
    - HPC Cluster: NSCC ASPIRE2A
    - Job Scheduler: PBS Pro

===============================================================================
3. BENCHMARKS INCLUDED
===============================================================================

Benchmark 1: Warp Divergence (src/warp_divergence.cu)
---------------------------------------------------------
Measures throughput degradation when threads within a warp execute different
control flow paths. Warps execute 32 threads in lockstep — divergence forces
serialization.

    Uniform Kernel:    All threads execute same branch
    Divergent Kernel:  50% threads take one branch, 50% take another

Benchmark 2: Memory Bank Conflicts (src/memory_bank.cu)
---------------------------------------------------------
Analyzes shared memory access patterns and their impact on latency. A100 has
32 shared memory banks — conflicts occur when multiple threads access the
same bank simultaneously.

    Stride 1:   Threads access Bank 0,1,2,3... (no conflict)
    Stride 32:  All threads access Bank 0,0,0,0... (max conflict)

Benchmark 3: Kernel Occupancy (src/occupancy.cu)
---------------------------------------------------------
Tests different block sizes to find optimal throughput. More threads does not
always equal faster execution due to register pressure and resource limits.

    Block Sizes Tested: 128, 256, 512, 1024 threads per block

===============================================================================
4. RESULTS SUMMARY
===============================================================================

+------------------------+------------------+------------------+-------------+
| Benchmark              | Best Case        | Worst Case       | Degradation |
+------------------------+------------------+------------------+-------------+
| Warp Divergence        | 998.43 ms        | 1990.37 ms       | 1.99x       |
+------------------------+------------------+------------------+-------------+
| Memory Bank (Stride)   | 34.48 ms (2)     | 545.49 ms (32)   | 15.8x       |
+------------------------+------------------+------------------+-------------+
| Occupancy (Block Size) | 0.0051 ms (512)  | 1.6763 ms (128)  | 328x        |
+------------------------+------------------+------------------+-------------+

Key Findings:

    1. Warp divergence causes ~2x slowdown on A100 when branches split 50/50.
       This is lower than older GPUs due to Ampere's independent thread
       scheduling feature.

    2. Memory bank conflicts cause 15.8x latency spikes. Stride-32 access
       patterns should be avoided in shared memory kernels.

    3. Block 512 achieved optimal occupancy. Block 1024 was slower due to
       register pressure causing spills to slower memory.

===============================================================================
5. TECHNICAL DEEP DIVE
===============================================================================

Warp Divergence Explained:
--------------------------
A warp consists of 32 threads that execute in SIMT (Single Instruction,
Multiple Threads) fashion. When threads diverge:

    Step 1: Warp executes Path A for threads where condition is true
    Step 2: Threads on Path B wait (inactive)
    Step 3: Warp executes Path B for remaining threads
    Step 4: Threads on Path A wait (inactive)

This serialization is why divergent kernels took 1990ms vs 998ms uniform.

Memory Banking Explained:
-------------------------
A100 shared memory is divided into 32 banks (4 bytes each). When multiple
threads access addresses in the same bank:

    No Conflict:  Bank 0, Bank 1, Bank 2, Bank 3... (parallel access)
    Conflict:     Bank 0, Bank 0, Bank 0, Bank 0... (serialized access)

Our Stride-32 test forced all 256 threads into 8 bank conflicts per warp,
resulting in 545ms vs 34ms for conflict-free access.

Occupancy Explained:
--------------------
Occupancy = (Active Warps / Maximum Possible Warps) x 100%

Higher occupancy helps hide memory latency, but:

    - Too few threads (Block 128): Under-utilized SM resources
    - Optimal (Block 512): Balanced registers, memory, and compute
    - Too many (Block 1024): Register spilling to local memory (slow)

===============================================================================
6. ASPIRE2A HPC DEPLOYMENT
===============================================================================

PBS Job Script Configuration:
-----------------------------
The project runs on ASPIRE2A via PBS Pro scheduler with enforced GPU ratios:

    #PBS -N warp_arch_explorer
    #PBS -l select=1:ncpus=16:mem=110G:ngpus=1
    #PBS -l walltime=01:00:00
    #PBS -q normal
    #PBS -P <PROJECT_ID>

Resource Ratio (Enforced by ASPIRE2A):
    - 1 GPU = 16 CPUs + 110 GB RAM
    - This ensures fair resource allocation across users

Module Environment:
-------------------
    module load cuda/12.2.2
    module load miniforge3
    conda activate test_ai_amd

Compilation:
------------
    nvcc -o bin/warp_divergence src/warp_divergence.cu
    nvcc -o bin/memory_bank src/memory_bank.cu
    nvcc -o bin/occupancy src/occupancy.cu

===============================================================================
7. QUICK START GUIDE
===============================================================================

Step 1: Clone Repository
------------------------
    git clone https://github.com/yourusername/cuda-warp-architecture-explorer.git
    cd cuda-warp-architecture-explorer

Step 2: Submit PBS Job (ASPIRE2A)
---------------------------------
    qsub run_benchmarks.pbs

Step 3: Monitor Job Status
--------------------------
    qstat -u <your_username>

Step 4: View Results
--------------------
    cat logs/warp_output.txt
    python src/analyze_results.py

Step 5: Check Generated Plots
-----------------------------
    ls results/
    # Output: plot_divergence.png, plot_memory.png, plot_occupancy.png

===============================================================================
8. REPOSITORY STRUCTURE
===============================================================================

cuda-warp-architecture-explorer/
    ├── src/
    │   ├── warp_divergence.cu      # Warp divergence benchmark kernel
    │   ├── memory_bank.cu          # Shared memory bank conflict kernel
    │   ├── occupancy.cu            # Block size occupancy kernel
    │   └── analyze_results.py      # Python analysis and plotting
    ├── bin/
    │   ├── warp_divergence         # Compiled divergence binary
    │   ├── memory_bank             # Compiled memory bank binary
    │   └── occupancy               # Compiled occupancy binary
    ├── results/
    │   ├── warp_divergence.csv     # Timing results (divergence)
    │   ├── memory_bank.csv         # Timing results (banking)
    │   ├── occupancy.csv           # Timing results (occupancy)
    │   ├── plot_divergence.png     # Visualization (divergence)
    │   ├── plot_memory.png         # Visualization (banking)
    │   └── plot_occupancy.png      # Visualization (occupancy)
    ├── logs/
    │   └── warp_output.txt         # PBS job output log
    ├── run_benchmarks.pbs          # PBS job submission script
    └── README.md                   # This file

===============================================================================
9. KEY LEARNINGS
===============================================================================

1. Compiler Optimization Can Hide Bottlenecks
   ---------------------------------------------
   Initial benchmarks showed INVERTED results (divergence was faster).
   The compiler was optimizing away the very patterns we wanted to measure.
   Fix: Used volatile __shared__ and equalized compute intensity.

2. A100 Architecture Differs from Textbook Examples
   -------------------------------------------------
   Ampere's Independent Thread Scheduling reduces divergence penalties
   compared to Volta/Tesla. Expected 5-10x slowdown, measured ~2x.

3. Occupancy Does Not Equal Performance
   -------------------------------------
   Block 512 outperformed Block 1024 despite lower theoretical occupancy.
   Register pressure and memory bandwidth are better optimization targets.

4. Real Profiling Requires Careful Kernel Design
   ----------------------------------------------
   Timing must account for:
       - Warmup runs (first kernel launch is slower)
       - Multiple iterations (average over 10+ runs)
       - CUDA event synchronization (avoid measuring async overhead)

===============================================================================
10. LICENSE
===============================================================================

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

===============================================================================
                              END OF README
===============================================================================
