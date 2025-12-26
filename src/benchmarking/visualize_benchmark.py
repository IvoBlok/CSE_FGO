import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import rcParams
import os

def parse_command_line():
    """Parse command line arguments."""
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <benchmark_results.json>")
        sys.exit(1)
    
    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"Error: File '{json_file}' not found!")
        sys.exit(1)
    
    return json_file

def load_benchmark_data(json_file):
    """Load and parse the benchmark results."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded benchmark data from: {json_file}")
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Total benchmark time: {data.get('totalBenchmarkTime', 0) / 1e6:.2f} seconds")
    print(f"Number of N values: {len(data.get('allNResults', []))}")
    
    return data

def extract_plot_data(data):
    """Extract data for plotting and exact solutions."""
    plot_data = {}
    
    for n_result in data.get('allNResults', []):
        n_data = {
            'total': [],
            'dmc': [],
            'real_opt': [],
            'other': [],
            'energies': [],
            'exact': None
        }

        multi_run_result = n_result.get('multiRunResult', {})
        n = multi_run_result.get('n', 0)
        
        n_data['exact'] = n_result.get('exactSolution')

        for single_run in multi_run_result.get('allRuns', []):
            total = single_run.get('totalTime') / 1e6
            dmc = single_run.get('dmcTime') / 1e6
            real_opt = single_run.get('realOptTime') / 1e6

            n_data['total'].append(total)
            n_data['dmc'].append(dmc)
            n_data['real_opt'].append(real_opt)
            n_data['other'].append(total - (dmc + real_opt))

            for cluster in single_run.get('candidates', []):
                n_data['energies'].append(cluster[1])

        for key in ['total', 'dmc', 'real_opt', 'other', 'energies']:
            n_data[key] = np.array(n_data[key])
            
        plot_data[n] = n_data

    return plot_data

def create_timing_plot(ax, plot_data):
    n_values = sorted(plot_data.keys())

    total_time = 0
    dmc_avgs = []
    real_opt_avgs = []
    other_avgs = []

    for n in n_values:
        data = plot_data[n]

        total_time += np.sum(data['total'])
        avg_dmc = np.mean(data['dmc'])
        avg_real_opt = np.mean(data['real_opt'])
        avg_other = np.mean(data['other'])

        dmc_avgs.append(avg_dmc)
        real_opt_avgs.append(avg_real_opt)
        other_avgs.append(avg_other)

    dmc_avgs = np.array(dmc_avgs)
    real_opt_avgs = np.array(real_opt_avgs)
    other_avgs = np.array(other_avgs)

    x = np.arange(len(n_values))
    width = 0.6

    other_bars = ax.bar(x, real_opt_avgs + dmc_avgs + other_avgs, width, label='Other', color='lightgreen', edgecolor='black')
    dmc_bars = ax.bar(x, real_opt_avgs + dmc_avgs, width, label='DMC', color='steelblue', edgecolor='black')
    real_opt_bars = ax.bar(x, real_opt_avgs, width, label='realOpt', color='lightcoral', edgecolor='black')

    ax.set_xlabel('N')
    ax.set_ylabel('Average time for Single Run [s]')
    ax.set_title(f'Time Breakdown by N, Total: {total_time:.1f}s')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    return ax
        
def create_energy_boxplot(ax, plot_data):
    n_values = sorted(plot_data.keys())
    width = 0.6

    energy_data = []
    exact_values = []

    for n in n_values:
        data = plot_data[n]
        energies = data['energies']
        exact = data['exact']

        energy_data.append(energies)
        exact_values.append(exact)

    x = np.arange(len(n_values))
    box = ax.boxplot(energy_data, positions=x, widths=width, patch_artist=True, whis=[0, 100])

    for box_element in box['boxes']:
        box_element.set_facecolor('lightblue')
        box_element.set_edgecolor('black')
    
    for median in box['medians']:
        median.set_color('darkblue')
        median.set_linewidth(2)
    
    for whisker in box['whiskers']:
        whisker.set_color('black')
    
    for cap in box['caps']:
        cap.set_color('black')
    
    for flier in box['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('red')
        flier.set_markersize(5)

    for i, (n, exact) in enumerate(zip(n_values, exact_values)):
        if exact is not None:
            # Draw line across the entire boxplot
            ax.hlines(y=exact, xmin=i - width/2, xmax=i + width/2, 
                     color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_ylim([None, 0])
    ax.set_xlabel('N')
    ax.set_ylabel('Energy [-]')
    ax.set_title('Energy Distribution of Candidates by N')
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.grid(True, alpha=0.3, axis='y')

    return ax
    


def main():
    json_file = parse_command_line()
    data = load_benchmark_data(json_file)
    plot_data = extract_plot_data(data)
    
    if not plot_data.keys():
        print("Error: No valid data found in JSON file!")
        sys.exit(1)
    
    rcParams.update({
        'figure.autolayout': True,
        'figure.figsize': (14, 6),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'DejaVu Sans'
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    
    ax1 = create_timing_plot(ax1, plot_data)
    ax2 = create_energy_boxplot(ax2, plot_data)
    
    fig.suptitle(f'FGO Benchmark Results - {data.get("timestamp", "")}', 
                 fontsize=16, fontweight='bold')
    
    # Save the figure
    base_name = os.path.splitext(json_file)[0]
    output_file = f"{base_name}_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_file}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()