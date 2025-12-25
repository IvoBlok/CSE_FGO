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
    plot_data = {
        'n_values': [],
        'times': {
            'total': [],
            'dmc': [],
            'realOpt': [],
            'other': []  # total - (dmc + realOpt)
        },
        'discrete_energies': [],  # List of lists, one per N
        'continuous_energies': []  # List of lists, one per N
    }
    
    exact_solutions = {}
    
    for n_result in data.get('allNResults', []):
        multi_run_result = n_result.get('multiRunResult', {})
        n = multi_run_result.get('n', 0)
        
        if n == 0:
            continue  # Skip invalid entries
        
        # Store exact solution for this N
        exact_solutions[n] = n_result.get('exactSolution', float('nan'))
        
        plot_data['n_values'].append(n)
        
        # Extract timing data from first run (assuming all runs have similar breakdown)
        all_runs = multi_run_result.get('allRuns', [])
        if all_runs:
            first_run = all_runs[0]
            total_time = first_run.get('totalTime', 0) / 1e6  # Convert to seconds
            dmc_time = first_run.get('dmcTime', 0) / 1e6
            real_opt_time = first_run.get('realOptTime', 0) / 1e6
            other_time = max(0, total_time - dmc_time - real_opt_time)
            
            plot_data['times']['total'].append(total_time)
            plot_data['times']['dmc'].append(dmc_time)
            plot_data['times']['realOpt'].append(real_opt_time)
            plot_data['times']['other'].append(other_time)
        
        # Extract energy data from all runs
        discrete_energies_n = []
        continuous_energies_n = []
        
        for run in all_runs:
            # Best discrete energy from discrete candidates
            discrete_candidates = run.get('discreteCandidates', [])
            if discrete_candidates:
                # Get minimum energy from discrete candidates
                discrete_energies = [cand[1] for cand in discrete_candidates if len(cand) > 1]
                if discrete_energies:
                    discrete_energies_n.append(min(discrete_energies))
            
            # Best continuous energy
            best_energy = run.get('bestEnergy', float('inf'))
            continuous_energies_n.append(best_energy)
        
        plot_data['discrete_energies'].append(discrete_energies_n)
        plot_data['continuous_energies'].append(continuous_energies_n)
    
    return plot_data, exact_solutions

def create_timing_plot(ax, plot_data):
    """Create the stacked bar chart for timing breakdown."""
    n_values = plot_data['n_values']
    dmc_times = plot_data['times']['dmc']
    real_opt_times = plot_data['times']['realOpt']
    other_times = plot_data['times']['other']
    
    # Create stacked bars
    bar_width = 0.6
    indices = np.arange(len(n_values))
    
    bars1 = ax.bar(indices, dmc_times, bar_width, 
                   label='DMC Time', color='#1f77b4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(indices, real_opt_times, bar_width, 
                   bottom=dmc_times, label='Real Opt Time', 
                   color='#ff7f0e', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(indices, other_times, bar_width, 
                   bottom=np.array(dmc_times) + np.array(real_opt_times),
                   label='Other Time', color='#2ca02c', edgecolor='black', linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Number of Atoms (N)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Average Time Breakdown per N', fontsize=14, fontweight='bold')
    
    # Set x-ticks to show N values
    ax.set_xticks(indices)
    ax.set_xticklabels(n_values, rotation=45, fontsize=10)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add value labels on top of bars for total time
    for i, (total, dmc, real_opt) in enumerate(zip(plot_data['times']['total'], 
                                                   dmc_times, real_opt_times)):
        if total > 0.1:  # Only label if time is significant
            ax.text(i, total + 0.02 * max(plot_data['times']['total']), 
                   f'{total:.2f}s', ha='center', va='bottom', fontsize=8)
    
    return ax

def create_energy_boxplot(ax, plot_data, exact_solutions):
    """Create boxplots for discrete and continuous energies."""
    n_values = plot_data['n_values']
    discrete_energies = plot_data['discrete_energies']
    continuous_energies = plot_data['continuous_energies']
    
    # Prepare data for boxplot
    positions = []
    boxplot_data = []
    labels = []
    
    # Create alternating positions for discrete and continuous boxplots
    for i, n in enumerate(n_values):
        # Discrete energies box
        if discrete_energies[i]:  # Check if we have data
            positions.append(i - 0.15)
            boxplot_data.append(discrete_energies[i])
            labels.append('')
        
        # Continuous energies box
        if continuous_energies[i]:  # Check if we have data
            positions.append(i + 0.15)
            boxplot_data.append(continuous_energies[i])
            labels.append('')
    
    # Create boxplot with custom properties
    boxprops = dict(linewidth=1.5, color='black')
    medianprops = dict(linewidth=1.5, color='black')  # Changed from red to black
    whiskerprops = dict(linewidth=1.5, color='black')
    capprops = dict(linewidth=1.5, color='black')
    
    # Create the boxplot with no fliers (no points)
    bp = ax.boxplot(boxplot_data, positions=positions, widths=0.2,
                    patch_artist=True, showfliers=False,  # No points
                    showmeans=False,  # Don't show mean markers
                    boxprops=boxprops, medianprops=medianprops,
                    whiskerprops=whiskerprops, capprops=capprops,
                    whis=[0, 100])  # Set whiskers to min and max
    
    # Color the boxes: discrete = blue, continuous = orange
    colors = []
    for i in range(len(positions)):
        if i % 2 == 0:  # Even indices are discrete
            colors.append('#1f77b4')
        else:  # Odd indices are continuous
            colors.append('#ff7f0e')
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add horizontal lines at exact solutions for each N
    for i, n in enumerate(n_values):
        if n in exact_solutions:
            exact_energy = exact_solutions[n]
            # Draw line segments at the exact x-positions of the boxplots
            # Discrete box position
            disc_pos = i - 0.15
            cont_pos = i + 0.15
            box_width = 0.2
            half_width = box_width / 2
            
            # Create two line segments: one for discrete, one for continuous
            # Discrete segment
            ax.hlines(y=exact_energy, 
                     xmin=disc_pos - half_width, 
                     xmax=disc_pos + half_width,
                     color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Continuous segment
            ax.hlines(y=exact_energy, 
                     xmin=cont_pos - half_width, 
                     xmax=cont_pos + half_width,
                     color='red', linestyle='--', alpha=0.7, linewidth=1)
            
            # Add a connecting line between the two segments (optional)
            ax.hlines(y=exact_energy, 
                     xmin=disc_pos + half_width, 
                     xmax=cont_pos - half_width,
                     color='red', linestyle=':', alpha=0.3, linewidth=0.5)
    
    # Customize the plot
    ax.set_xlabel('Number of Atoms (N)', fontsize=12)
    ax.set_ylabel('Energy', fontsize=12)
    ax.set_title('Energy Distribution: Discrete vs Continuous', 
                 fontsize=14, fontweight='bold')
    
    # Set x-ticks at N values
    ax.set_xticks(range(len(n_values)))
    ax.set_xticklabels(n_values, rotation=45, fontsize=10)
    
    # Set x-axis limits to include all boxplots with padding
    ax.set_xlim(-0.5, len(n_values) - 0.5)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Create custom legend
    discrete_patch = mpatches.Patch(color='#1f77b4', alpha=0.7, label='Discrete Best Energies')
    continuous_patch = mpatches.Patch(color='#ff7f0e', alpha=0.7, label='Continuous Best Energies')
    exact_line = mpatches.Patch(color='red', alpha=0.7, linestyle='--', linewidth=1, 
                                fill=False, label='Exact Solution')
    ax.legend(handles=[discrete_patch, continuous_patch, exact_line], 
              loc='upper right', fontsize=10)
    
    return ax, bp

def main():
    """Main function to create plots."""
    # Parse command line
    json_file = parse_command_line()
    
    # Load data
    data = load_benchmark_data(json_file)
    
    # Extract plot data and exact solutions
    plot_data, exact_solutions = extract_plot_data(data)
    
    if not plot_data['n_values']:
        print("Error: No valid data found in JSON file!")
        sys.exit(1)
    
    # Set up matplotlib style
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
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create timing plot
    print("\nCreating timing breakdown plot...")
    ax1 = create_timing_plot(ax1, plot_data)
    
    # Create energy boxplot
    print("Creating energy distribution boxplot...")
    ax2, boxplot = create_energy_boxplot(ax2, plot_data, exact_solutions)
    
    # Add overall title
    fig.suptitle(f'FGO Benchmark Results - {data.get("timestamp", "")}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Save the figure
    base_name = os.path.splitext(json_file)[0]
    output_file = f"{base_name}_analysis.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to: {output_file}")
    
    # Display summary statistics
    print("\n=== Summary Statistics ===")
    print(f"{'N':>3} | {'Exact Energy':>12} | {'Discrete Min':>12} {'Max':>12} {'Avg':>12} | {'Continuous Min':>12} {'Max':>12} {'Avg':>12}")
    print("-" * 120)
    
    for i, n in enumerate(plot_data['n_values']):
        disc_energies = plot_data['discrete_energies'][i]
        cont_energies = plot_data['continuous_energies'][i]
        exact_energy = exact_solutions.get(n, float('nan'))
        
        if disc_energies and cont_energies:
            disc_min = min(disc_energies)
            disc_max = max(disc_energies)
            disc_avg = np.mean(disc_energies)
            
            cont_min = min(cont_energies)
            cont_max = max(cont_energies)
            cont_avg = np.mean(cont_energies)
            
            print(f"{n:3d} | {exact_energy:12.6f} | "
                  f"{disc_min:12.6f} {disc_max:12.6f} {disc_avg:12.6f} | "
                  f"{cont_min:12.6f} {cont_max:12.6f} {cont_avg:12.6f}")
    
    # Show the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()