import re
import ast
import stat_pareto  # To import the pareto front data

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.cm as cm

def parse_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    generations = []
    pred_tec_means = []
    pred_density_means = []
    pred_density_stds = []
    pred_tec_stds = []
    targets = []
    best_individuals = {}

    # Define regex patterns
    gen_pattern = re.compile(r"- Generation (\d+)")
    tec_pattern = re.compile(r"pred_tec_mean:\s*([-+]?\d*\.\d+|\d+)")
    tec_std_pattern = re.compile(r"tec_std:\s*([-+]?\d*\.\d+|\d+)")
    density_pattern = re.compile(r"pred_density_mean:\s*([-+]?\d*\.\d+|\d+)")
    density_std_pattern = re.compile(r"density_std:\s*([-+]?\d*\.\d+|\d+)")
    target_pattern = re.compile(r"target:\s*([-+]?\d*\.\d+|\d+)")

    read_blocks = False
    best_individual_str = ""
    for line in lines:
        if "Elements: " in line:
            element_list = ast.literal_eval(line.split("Elements: ")[-1])
        # Check for section start
        if "====" in line:
            read_blocks = True
            continue
        
        # Check for section end
        if "----" in line:
            read_blocks = False
            continue
        
        # Only process lines within the relevant section
        if read_blocks:
            # Match Generation
            gen_match = gen_pattern.search(line)
            if gen_match:
                generations.append(int(gen_match.group(1)))
            
            # Match pred_tec_mean
            tec_match = tec_pattern.search(line)
            if tec_match:
                pred_tec_means.append(float(tec_match.group(1)))
            
            # Match pred_density_mean
            density_match = density_pattern.search(line)
            if density_match:
                pred_density_means.append(float(density_match.group(1)))
            
            # Match tec_std
            
            tec_std_match = tec_std_pattern.search(line)
            if tec_std_match:
                pred_tec_stds.append(float(tec_std_match.group(1)))
            
            # Match density_std
            density_std_match = density_std_pattern.search(line)
            if density_std_match:
                pred_density_stds.append(float(density_std_match.group(1)))

            
            # Match target
            target_match = target_pattern.search(line)
            if target_match:
                targets.append(float(target_match.group(1)))


        if "Best Individual: [" in line and not "]" in line:
            best_individual_str = line.split("Best Individual: [")[-1]  # 获取列表的第一部分
        
        elif "]" in line and best_individual_str:
            best_individual_str += line.split("]")[0] 
            best_individual = list(map(float, best_individual_str.split()))
            for idx, e in enumerate(element_list):
                best_individuals[e] = best_individual[idx]
            
            best_individual_str = "" 
        elif "Best Individual: [" in line and '[' in line and ']' in line:
            
            s = line.split("Best Individual: [")[-1].split("]")[0]
            best_individual = list(map(float, s.split()))
            for idx, e in enumerate(element_list):
                best_individuals[e] = best_individual[idx]
    return generations, pred_tec_means, pred_density_means, targets, pred_tec_stds, pred_density_stds, best_individuals


def plot_ga(g, tec, density, target, pred_tec_stds, pred_density_stds):
    # Convert inputs to numpy arrays for convenience
    g = np.array(g)
    tec = np.array(tec)
    density = np.array(density)
    target = np.array(target)
    pred_tec_stds = np.array(pred_tec_stds)
    pred_density_stds = np.array(pred_density_stds)

    # Subplot 1: TEC over generations
    plt.subplot(141)
    plt.plot(g, tec)
    plt.fill_between(g, tec - pred_tec_stds, tec + pred_tec_stds, alpha=0.2)
    plt.ylabel("TEC")
    
    # Inset for TEC standard deviation
    ax_inset_tec = inset_axes(plt.gca(), width="30%", height="30%", loc="upper right")
    ax_inset_tec.plot(g, pred_tec_stds, color='orange', label="TEC STD")
    ax_inset_tec.set_ylabel("STD")
    
    # Subplot 2: Density over generations
    plt.subplot(142)
    plt.plot(g, density)
    plt.fill_between(g, density - pred_density_stds, density + pred_density_stds, alpha=0.2)
    plt.ylabel("Density")
    plt.xlabel("Generation")
    
    # Inset for Density standard deviation
    ax_inset_density = inset_axes(plt.gca(), width="30%", height="30%", loc="upper right")
    ax_inset_density.plot(g, pred_density_stds, color='orange')
    ax_inset_density.set_ylabel("STD")
    
    # Subplot 3: Target over generations
    plt.subplot(143)
    plt.plot(g, target)
    plt.ylabel("Target")

    plt.subplot(144)
    plt.xlabel("TEC")
    plt.ylabel("Density")
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    raw_tec, raw_den, pareto_front, _ = stat_pareto.get_pareto_front(data_file)
    
    # Plot the original Pareto front
    plt.scatter(raw_tec, raw_den, c='#9999AA', label="Orig. Data", alpha=0.5)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='#882233', label="Orig. Pareto Front.", marker='s')
    
    # Use the new method to get the colormap
    cmap = plt.get_cmap('viridis')
    
    # Create a ScalarMappable object with a dummy array for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=g.min(), vmax=g.max()))
    sm.set_array([])  # Provide an empty array for the ScalarMappable object
    
    # Get the current axis
    ax = plt.gca()
    
    # Add a colorbar to the current axis
    cbar = ax.figure.colorbar(sm, ax=ax, label="Generation")
    
    # Plot the new points with a color mapping to the generation
    sc = ax.scatter(tec, density, c=g, label="New Points", marker='x', cmap=cmap)
    ax.legend(loc="upper right", fontsize=8)



if __name__ == "__main__":
    # Parse the log file
    import glob
    import stat_pareto

    logs = glob.glob("20241209*.log")
    for log in logs:
        print(f"=========Parsing {log}")
        generations, pred_tec_means, pred_density_means, targets, pred_tec_stds, pred_density_stds, best_individuals = parse_log(log)
        log_name = log.split(".log")[0]
        # Print results
        print("Generations:", generations)
        print("Pred Tec Means:", pred_tec_means)
        print("Pred Density Means:", pred_density_means)
        print("Best Individuals:", best_individuals)
        print("Targets:", targets)

        try:
            plt.figure(figsize=(16,4))
            plt.rcParams['font.size'] = 12
            plt.rcParams['font.family'] = 'Arial'
            plot_ga(generations, pred_tec_means, pred_density_means, targets, pred_tec_stds, pred_density_stds)
            
            plt.tight_layout(pad=0.2)
            plt.savefig(f"{log_name}.png", dpi=300)
        except Exception as e:
            print(f"Error plotting {log}: {e}")
