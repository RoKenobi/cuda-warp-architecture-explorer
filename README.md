# GPU Reliability & Architecture Analysis Platform

A comprehensive suite of 3 projects demonstrating GPU computing, machine learning, 
and high-performance computing (HPC) on the NSCC ASPIRE2A supercomputer with 
NVIDIA A100 GPUs.

## Overview

Modern GPU failures rarely occur as isolated events. Hardware instability emerges 
through cascading subsystem interactions:

    Voltage Instability → Thermal Rise → Memory Pressure → GPU Crash

This repository contains three distinct projects that together form a complete 
GPU reliability and architecture analysis platform:

1. HPC CUDA Performance Profiler - Kernel optimization and bottleneck analysis
2. GPU Failure Simulation & ML Diagnostic Platform - Automated failure diagnosis
3. CUDA Warp Architecture Explorer - Low-level GPU execution model analysis

All projects were executed on ASPIRE2A using PBS job scheduling with proper 
GPU resource allocation (1 GPU : 16 CPU : 110GB RAM).

## Project Structure

    gpu-reliability-platform/
    ├── project-1-cuda-profiler/
    │   ├── src/
    │   │   ├── kernel_benchmarks.cu
    │   │   └── analyze_performance.py
    │   ├── run_profiler.pbs
    │   └── results/
    ├── project-2-failure-ml/
    │   ├── src/
    │   │   ├── cuda_simulator.py
    │   │   ├── graph_analyzer.py
    │   │   ├── ml_supervised.py
    │   │   └── ml_anomaly.py
    │   ├── run_gpu_pipeline.pbs
    │   ├── data/
    │   ├── models/
    │   └── logs/
    └── project-3-warp-architecture/
        ├── src/
        │   ├── warp_divergence.cu
        │   ├── memory_bank.cu
        │   ├── occupancy.cu
        │   └── analyze_results.py
        ├── run_benchmarks.pbs
        └── results/

## Project 1: HPC CUDA Performance Profiler

### Objective
Identify performance bottlenecks in custom CUDA kernels processing 256M+ elements.

### Key Findings
    Metric                      Result
    Uncoalesced Memory Access   5.9× performance degradation
    Atomic Contention           5.5× slowdown
    Kernel Launches Tracked     3,100+

### Tools Used
    - Nsight Systems for profiling
    - CUDA Events for nanosecond timing
    - Python (Pandas + Plotly) for telemetry visualization

### PBS Configuration
    #PBS -l select=1:ncpus=16:mem=110G:ngpus=1
    #PBS -l walltime=02:00:00
    #PBS -q normal

## Project 2: GPU Failure Simulation & ML Diagnostic Platform

### Objective
Simulate GPU telemetry data and train ML models to diagnose hardware failures.

### Architecture

    CUDA Simulation (Numba)
            ↓
    1M+ Telemetry Records
            ↓
    Graph Analysis (NetworkX)
            ↓
    Supervised ML (XGBoost + RF)
            ↓
    Unsupervised Anomaly Detection (Isolation Forest)

### Components

#### Component 1: CUDA Failure Simulation
    - Parallel telemetry generation using Numba CUDA kernels
    - 4 failure classes: Stable, Power Instability, Thermal Throttling, Memory Leak
    - 15% failure injection rate (150K failures from 1M events)

#### Component 2: Graph-Based Root Cause Analysis
    - Models subsystem dependencies (Power → Voltage → Thermal → GPU Core)
    - Calculates degree centrality to identify critical failure nodes
    - GPU Core identified as highest centrality (0.8000)

#### Component 3: Supervised ML Classification
    - Random Forest baseline + XGBoost industry standard
    - 99% F1-score across all 4 failure classes
    - Multi-core training using all 16 allocated CPUs

#### Component 4: Unsupervised Anomaly Detection
    - Isolation Forest for zero-day anomaly detection
    - No labels required - detects unknown failure patterns
    - 150,000 anomalous states identified

### Results

    Classification Report (XGBoost):
                        precision    recall  f1-score   support
    Stable                 1.00      1.00      1.00    170032
    Power Instability      0.99      0.98      0.99      9936
    Thermal Throttling     0.99      0.98      0.99     10035
    Memory Leak            1.00      1.00      1.00      9997
    accuracy                                       1.00    200000
    macro avg              0.99      0.99      0.99    200000

    Anomaly Detection:
    Normal States:    850,000
    Anomalous States: 150,000

### ASPIRE2A Compliance
    - All jobs submitted via PBS (no login node compute)
    - Data stored in /scratch (100TB) to avoid /home quota (50GB)
    - Module environment: miniforge3 + cuda/12.2.2
    - Conda environment: test_ai_amd

## Project 3: CUDA Warp Architecture Explorer

### Objective
Explore low-level GPU execution mechanics: warps, memory banks, and occupancy.

### Benchmarks

#### 1. Warp Divergence
    Uniform Kernel:    998.43 ms
    Divergent Kernel:  1990.37 ms
    Degradation:       1.99× (2× slowdown)

    Explanation: When threads in a warp take different branches, execution 
    serializes. A100 warps contain 32 threads executing in lockstep.

#### 2. Shared Memory Bank Conflicts
    Stride   Time (ms)   Bank Conflict Level
    1        41.40       None (optimal)
    2        34.48       Minimal
    4        68.33       Low
    8        136.38      Medium
    16       272.72      High
    32       545.49      Maximum (13× slower than stride 2)

    Explanation: A100 has 32 shared memory banks. Stride 32 causes all threads 
    to access the same bank simultaneously, creating request queuing.

#### 3. Kernel Occupancy Tuning
    Block Size   Time (ms)   Occupancy Level
    128          1.6763      Under-utilized
    256          0.0061      Good
    512          0.0051      Optimal (sweet spot)
    1024         0.0061      Register pressure

    Explanation: Block 512 achieved optimal throughput by balancing thread count 
    with register availability. Block 1024 caused register spilling.

### Key Learnings
    1. Compiler optimizations can mask bottlenecks - use volatile for memory tests
    2. A100 Ampere architecture has independent thread scheduling (reduces divergence)
    3. Occupancy does not equal performance - find the sweet spot empirically
    4. Timer resolution matters - run kernels multiple times for measurable results

## Technical Specifications

### Hardware
    Cluster:        NSCC ASPIRE2A
    GPU:            NVIDIA A100-SXM4-40GB
    CPU:            AMD EPYC 7713 (64 cores per node)
    RAM:            512 GB per node
    Storage:        100 TB /scratch, 50 GB /home

### Software
    Languages:      Python 3, C++/CUDA
    ML Libraries:   Scikit-learn, XGBoost, Pandas, NumPy
    CUDA:           Numba (Python), nvcc (C++)
    Graph:          NetworkX, Matplotlib
    Profiling:      Nsight Systems, CUDA Events
    Scheduler:      PBS Pro

### PBS Job Configuration (All Projects)
    #PBS -N <job_name>
    #PBS -l select=1:ncpus=16:mem=110G:ngpus=1
    #PBS -l walltime=01:00:00
    #PBS -q normal
    #PBS -P <project_id>
    #PBS -o logs/output.txt
    #PBS -e logs/error.txt

## Quick Start

### Prerequisites
    1. ASPIRE2A account with GPU queue access
    2. Conda environment with required packages
    3. PBS job submission permissions

### Setup
    ssh <userid>@aspire2a.nscc.sg
    module load miniforge3
    conda activate test_ai_amd
    cd ~/AMDProjects/gpu-reliability-platform

### Run Project 2 (ML Pipeline)
    qsub project-2-failure-ml/run_gpu_pipeline.pbs
    qstat -u $USER  # Monitor job
    cat logs/pbs_output.txt  # View results

### Run Project 3 (Architecture Benchmarks)
    qsub project-3-warp-architecture/run_benchmarks.pbs
    python src/analyze_results.py  # Generate plots

## Results Summary

    Project    Focus              Key Metric                    Status
    1          Performance        5.9× memory bottleneck        Complete
    2          ML + Simulation    99% F1, 150K anomalies        Complete
    3          Architecture       2× divergence, 13× banking    Complete

## Why This Matters

GPU clusters power AI training, scientific computing, and cloud infrastructure. 
When GPUs fail:

    - Millions in compute resources go idle
    - Training jobs lose hours of progress
    - Production services experience downtime

This platform demonstrates how to:

    1. Simulate hardware failures at scale (CUDA parallel generation)
    2. Diagnose root causes automatically (ML classification)
    3. Detect unknown anomalies (unsupervised learning)
    4. Understand hardware internals (warp/memory/occupancy analysis)

## Future Work

    - Real telemetry ingestion from production GPU clusters
    - GPU-accelerated graph analytics using RAPIDS cuGraph
    - Distributed training across multiple A100 nodes
    - Integration with NVIDIA DCGM for real hardware monitoring
    - Visualization dashboard for failure propagation

## References

    - ASPIRE2A QuickStart Guide: NSCC Documentation
    - CUDA Programming Guide: NVIDIA Developer
    - XGBoost Documentation: https://xgboost.readthedocs.io
    - Numba CUDA Guide: https://numba.readthedocs.io

## License

MIT License - Feel free to use for educational and research purposes.

## Contact

For questions about this project or ASPIRE2A usage:
    - GitHub Issues: [Your Repo Issues]
    - NSCC Support: help@nscc.sg
    - Project Documentation: See individual project folders
