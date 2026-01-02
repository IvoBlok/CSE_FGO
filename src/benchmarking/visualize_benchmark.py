import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from matplotlib import rcParams
import matplotlib.patches as mpatches
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
            'dmc1': [],
            'dmc2': [],
            'real_opt': [],
            'other': [],
            'start_energies': [],
            'dmc_energies': [],
            'real_energies': [],
            'samplesize': None,
            'correct': None,
            'exact': None
        }

        multi_run_result = n_result.get('multiRunResult', [])
        
        n_data['exact'] = n_result.get('exactSolution')
        n_data['samplesize'] = len(multi_run_result)
        n_data['correct'] = 0

        for single_run in multi_run_result:
            total = single_run.get('totalTime') / 1e6
            dmc1 = single_run.get('dmc1Time') / 1e6
            dmc2 = single_run.get('dmc2Time') / 1e6
            real_opt = single_run.get('realOptTime') / 1e6

            n_data['total'].append(total)
            n_data['dmc1'].append(dmc1)
            n_data['dmc2'].append(dmc2)
            n_data['real_opt'].append(real_opt)
            n_data['other'].append(total - (dmc1 + dmc2 + real_opt))

            index = 0
            for energy in single_run.get('discCandidates', []):
                if (index == 0):
                    n_data['start_energies'].append(energy)
                else:
                    n_data['dmc_energies'].append(energy)
                index += 1
            for energy in single_run.get('contCandidates', []):
                n_data['real_energies'].append(energy)

            for energy in single_run.get('contCandidates', []):
                if (energy and energy < n_data['exact'] + 1e-3):
                    n_data['correct'] += 1
                    break

        for key in ['total', 'dmc1', 'dmc2', 'real_opt', 'other', 'dmc_energies', 'real_energies']:
            n_data[key] = np.array(n_data[key])
        
        n = n_result.get('n')
        plot_data[n] = n_data

    return plot_data

def create_timing_plot(ax, plot_data):
    n_values = sorted(plot_data.keys())

    total_time = 0
    dmc1_avgs = []
    dmc2_avgs = []
    real_opt_avgs = []
    other_avgs = []

    for n in n_values:
        data = plot_data[n]

        total_time += np.sum(data['total'])
        avg_dmc1 = np.mean(data['dmc1'])
        avg_dmc2 = np.mean(data['dmc2'])
        avg_real_opt = np.mean(data['real_opt'])
        avg_other = np.mean(data['other'])

        dmc1_avgs.append(avg_dmc1)
        dmc2_avgs.append(avg_dmc2)
        real_opt_avgs.append(avg_real_opt)
        other_avgs.append(avg_other)

    dmc1_avgs = np.array(dmc1_avgs)
    dmc2_avgs = np.array(dmc2_avgs)
    real_opt_avgs = np.array(real_opt_avgs)
    other_avgs = np.array(other_avgs)

    width = 0.6

    other_bars = ax.bar(n_values, real_opt_avgs + dmc1_avgs + dmc2_avgs + other_avgs, width, label='Other', color='lightgreen', edgecolor='black')
    dmc2_bars = ax.bar(n_values, real_opt_avgs + dmc1_avgs + dmc2_avgs, width, label='DMC2', color='blue', edgecolor='black')
    dmc1_bars = ax.bar(n_values, real_opt_avgs + dmc1_avgs, width, label='DMC1', color='steelblue', edgecolor='black')
    real_opt_bars = ax.bar(n_values, real_opt_avgs, width, label='realOpt', color='lightcoral', edgecolor='black')

    ax.set_xlabel('N')
    ax.set_ylabel('Average time for Single Run [s]')
    ax.set_title(f'Time Breakdown by N, Total: {total_time:.1f}s')
    
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=15, integer=True))

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    return ax
        
def create_energy_boxplot(ax, plot_data):
    n_values = sorted(plot_data.keys())
    width = 0.6 * min(np.diff(n_values)) if len(n_values) > 1 else 0.6

    real_energy_data = []
    dmc_energy_data = []
    start_energy_data = []
    exact_values = []
    correct_counts = []
    sample_sizes = []

    for n in n_values:
        data = plot_data[n]
        real_energies = data['real_energies']
        dmc_energies = data['dmc_energies']
        start_energies = data['start_energies']
        exact = data['exact']

        start_energy_data.append(start_energies)
        real_energy_data.append(real_energies)
        dmc_energy_data.append(dmc_energies)
        exact_values.append(exact)
        correct_counts.append(data['correct'])
        sample_sizes.append(data['samplesize'])

    DMCPlots = ax.violinplot(dmc_energy_data, positions=n_values, showmeans=False, showmedians=False, showextrema=False)
    RealPlots = ax.violinplot(real_energy_data, positions=n_values, showmeans=False, showmedians=False)

    for pc in DMCPlots['bodies']:
        pc.set_facecolor("#1DD7FC")
        pc.set_edgecolor("#000000D5")
        pc.set_alpha(0.3)

    for pc in RealPlots['bodies']:
        pc.set_facecolor("#001AAD")
        pc.set_edgecolor("#000000D5")
        pc.set_alpha(0.5)

    for line_component in ['cmins', 'cmaxes', 'cbars']:
        if line_component in DMCPlots:
            DMCPlots[line_component].set_color("#000000D5")
            DMCPlots[line_component].set_alpha(0.3)

    for line_component in ['cmins', 'cmaxes', 'cbars']:
        if line_component in RealPlots:
                RealPlots[line_component].set_color("#000000D5")
                RealPlots[line_component].set_alpha(0.5)

    for n, starts, exact, correct, samplesize in zip(n_values, start_energy_data, exact_values, correct_counts, sample_sizes):
        ax.hlines(y=exact, xmin=n - width/2, xmax=n + width/2, 
                    color="#960000", linestyle='--', linewidth=2, alpha=0.7)
        
        ax.text(n, exact - 1, correct, 
                ha='center', va='top',
                fontsize=8,
                color='grey',
                rotation=-90,
        )

        for start in starts:
            ax.hlines(y=start, xmin=n - width/3, xmax=n + width/3, 
                        color="#025D13", linestyle='--', linewidth=1, alpha=0.2)
    
    exact_values = np.array(exact_values)
    ax.set_ylim([np.min(exact) * 1.05, 0])
    ax.set_xlabel('N')
    ax.set_ylabel('Energy [-]')
    ax.set_title(f'Energy Distribution of Candidates by N, {sample_sizes[0]} samples')
    
    n_values = np.array(n_values)
    marked_n = np.linspace(np.min(n_values), np.max(n_values), 14, dtype=int)
    ax.set_xticks(marked_n)
    ax.set_xticklabels([str(n) for n in marked_n])

    start_patch = mpatches.Patch(color="#025D13", alpha=0.5, label='Start Clusters (Step 1)')
    dmc_patch = mpatches.Patch(color="#1DD7FC", alpha=0.3, label='DMC Clusters (Step 2)')
    real_patch = mpatches.Patch(color="#001AAD", alpha=0.5, label='Real Clusters (Step 3)')
    min_patch = mpatches.Patch(color="#960000", alpha=0.7, label='Global Minimum')

    ax.legend(handles=[start_patch, dmc_patch, real_patch, min_patch], loc='upper right')
    ax.grid(True, alpha=0.3)
    
    def format_coord(x, y):
            closest_n = min(n_values, key=lambda n_val: abs(n_val - x))
            return f"N={closest_n}, E={y:.1f}"
    ax.format_coord = format_coord

    return ax
    
def create_candidate_length_plot(ax, plot_data):
    n_values = sorted(plot_data.keys())

    avg_dmc_candidate_amount = []
    avg_real_candidate_amount = []

    for n in n_values:
        data = plot_data[n]
        avg_dmc_candidate_amount.append(len(data['dmc_energies']) / data['samplesize'])
        avg_real_candidate_amount.append(len(data['real_energies']) / data['samplesize'])

    ax.plot(n_values, avg_dmc_candidate_amount, label="avg #DMC candidates")
    ax.plot(n_values, avg_real_candidate_amount, label="avg #real candidates")

    ax.set_xlabel('N')
    ax.set_title("Number of DMC / real candidates by N")

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

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
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 10))
    
    ax1 = create_timing_plot(ax1, plot_data)
    ax2 = create_energy_boxplot(ax2, plot_data)
    ax3 = create_candidate_length_plot(ax3, plot_data)
    
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