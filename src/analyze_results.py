# src/analyze_results.py
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_results():
    print("Generating Analysis Plots...")
    
    # 1. Warp Divergence
    if os.path.exists('results/warp_divergence.csv'):
        df = pd.read_csv('results/warp_divergence.csv')
        plt.figure()
        plt.bar(df['Mode'], df['Time_ms'], color=['green', 'red'])
        plt.title('Warp Divergence Performance Penalty')
        plt.ylabel('Time (ms)')
        plt.savefig('results/plot_divergence.png')
        print("Saved plot_divergence.png")

    # 2. Memory Banking
    if os.path.exists('results/memory_bank.csv'):
        df = pd.read_csv('results/memory_bank.csv')
        plt.figure()
        plt.plot(df['Stride'], df['Time_ms'], marker='o')
        plt.title('Shared Memory Bank Conflicts')
        plt.xlabel('Access Stride')
        plt.ylabel('Time (ms)')
        plt.savefig('results/plot_memory.png')
        print("Saved plot_memory.png")

    # 3. Occupancy
    if os.path.exists('results/occupancy.csv'):
        df = pd.read_csv('results/occupancy.csv')
        plt.figure()
        plt.plot(df['BlockSize'], df['Time_ms'], marker='s', color='orange')
        plt.title('Kernel Occupancy vs Block Size')
        plt.xlabel('Block Size')
        plt.ylabel('Time (ms)')
        plt.savefig('results/plot_occupancy.png')
        print("Saved plot_occupancy.png")

if __name__ == "__main__":
    # Load conda env if running on compute node
    import subprocess
    subprocess.run(["module", "load", "miniforge3"], shell=True)
    plot_results()
