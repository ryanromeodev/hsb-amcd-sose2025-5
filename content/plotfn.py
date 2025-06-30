import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def wrap_phase(phase):
    return (phase + np.pi) % (2 * np.pi) - np.pi

def freqAnalysisPlot(title,graphsFilename,cols,seperator,invertPhaseGain):    
    filename = graphsFilename
    df = pd.read_csv(filename, sep=seperator, engine='python', usecols=cols)
    
    # Rename columns
    if invertPhaseGain:
        df.columns = ['Frequency', 'Phase', 'Gain']
    else:
        df.columns = ['Frequency', 'Gain', 'Phase']
    df = df.apply(pd.to_numeric, errors='coerce')

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Gain
    line1, = ax1.plot(df['Frequency'], df['Gain'], 'b-', label='Gain (dB)')
    # ax1.scatter(1000, max(df['Gain']),s=100,c="Green")
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Gain (dB)', color='b')
    ax1.set_xscale('log')
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot Phase on secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(df['Frequency'],df['Phase'], 'r--', label='Phase (°)')
    ax2.set_ylabel('Phase (°)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Title and layout
    plt.title(title)
    fig.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()

def transCientAnalysisPlot(title,idealGraphsFilename,cols):
    filename = idealGraphsFilename
    df = pd.read_csv(filename, sep=',', engine='python', usecols=cols)

    #BPF

    # Rename columns
    df.columns = ['time', 'input', 'output']
    df = df.apply(pd.to_numeric, errors='coerce')

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Gain
    line1, = ax1.plot(df['time'], df['input'], 'g-', label='Volts (V)')    
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('Gain (dB)', color='b')    
    ax1.tick_params(axis='y', labelcolor='b')

    # Plot Phase on secondary y-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(df['time'],df['output'], 'y-', label='Phase (°)')
    ax2.set_ylabel('Phase (°)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    # Combine legends from both axes
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')

    # Title and layout
    plt.title(title)
    fig.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()



