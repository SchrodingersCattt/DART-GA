import re
import matplotlib.pyplot as plt

def parse_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    generations = []
    pred_tec_means = []
    pred_density_means = []
    targets = []

    # Define regex patterns
    gen_pattern = re.compile(r"- Generation (\d+)")
    tec_pattern = re.compile(r"pred_tec_mean:\s*([-+]?\d*\.\d+|\d+)")
    density_pattern = re.compile(r"pred_density_mean:\s*([-+]?\d*\.\d+|\d+)")
    target_pattern = re.compile(r"target:\s*([-+]?\d*\.\d+|\d+)")

    read_blocks = False

    for line in lines:
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
            
            # Match target
            target_match = target_pattern.search(line)
            if target_match:
                targets.append(float(target_match.group(1)))

    return generations, pred_tec_means, pred_density_means, targets

def plot_ga(g, tec, density, target):
    plt.subplot(131)
    plt.plot(g, tec)
    plt.ylabel("TEC")

    plt.subplot(132)
    plt.plot(g, density)
    plt.ylabel("Density")
    plt.xlabel("Generation")

    plt.subplot(133)
    plt.plot(g, target)
    plt.ylabel("Target")

    plt.tight_layout()

if __name__ == "__main__":
    # Parse the log file
    import glob
    logs = glob.glob("20241125*.log")
    for log in logs:
        generations, pred_tec_means, pred_density_means, targets = parse_log(log)
        log_name = log.split(".log")[0]
        # Print results
        print("Generations:", generations)
        print("Pred Tec Means:", pred_tec_means)
        print("Pred Density Means:", pred_density_means)
        print("Targets:", targets)

        plt.figure(figsize=(8,3))
        plot_ga(generations, pred_tec_means, pred_density_means, targets)
        plt.savefig(f"ga_results_{log_name}.png")
