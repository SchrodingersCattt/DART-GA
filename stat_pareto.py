from stat_distribution import ExcelParser
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def is_dominated(target, others):
    for other in others:
        if all(o <= t for o, t in zip(other, target)) and any(o < t for o, t in zip(other, target)):
            return True
    return False

def extract_data(parser):
    tec = []
    density = []
    packing = []
    Fe, Ni, Co, Cr, V, Cu = [], [], [], [], [], []

    for row in tqdm(parser.iterate_rows()):
        tec.append(parser.get_tec(row))
        density.append(parser.get_density(row))
        Fe.append(parser.get_Fe(row))
        Ni.append(parser.get_Ni(row))
        Co.append(parser.get_Co(row))
        Cr.append(parser.get_Cr(row))
        V.append(parser.get_V(row))
        Cu.append(parser.get_Cu(row))
        packing.append((parser.get_phase(row)).replace("'", ""))

    return np.array(tec), np.array(density), np.array(Fe), np.array(Ni), np.array(Co), np.array(Cr), np.array(V), np.array(Cu), packing

def get_pareto_front(data_file, output=False):
    parser = ExcelParser(data_file)
    tec, density, Fe, Ni, Co, Cr, V, Cu, packing = extract_data(parser)

    points = np.column_stack((tec, density))

    pareto_indices = []
    for i, point in enumerate(points):
        if not is_dominated(point, np.delete(points, i, axis=0)):
            pareto_indices.append(i)
    if output:
        output_results(pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing)
    return tec, density, points[pareto_indices], pareto_indices

def output_results(pareto_indices, Fe, Ni, Co, Cr, V, Cu, packing):
    pareto_elements = {
        "Fe": np.array(Fe)[pareto_indices],
        "Ni": np.array(Ni)[pareto_indices],
        "Co": np.array(Co)[pareto_indices],
        "Cr": np.array(Cr)[pareto_indices],
        "V": np.array(V)[pareto_indices],
        "Cu": np.array(Cu)[pareto_indices],
        "packing": np.array(packing)[pareto_indices]
    }

    pareto_df = pd.DataFrame(pareto_elements)
    print(pareto_df.to_markdown())
    print([list(x[:-1]) for x in pareto_df.to_numpy()])

def plot_results(tec, density, pareto_front):
    plt.figure(figsize=(8,8))
    plt.rcParams['font.size'] = 16
    plt.scatter(tec, density, label="All Alloys", alpha=0.5)
    plt.scatter(pareto_front[:, 0], pareto_front[:, 1], c='r', label="Pareto Frontier")
    plt.xlabel("TEC")
    plt.ylabel("Density")
    plt.legend(loc="lower right")
    plt.savefig("pareto_front.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    data_file = "dataset/Data_base_DFT_Thermal.xlsx"
    
    tec, density, pareto_front, pareto_indices = get_pareto_front(data_file)
    
    plot_results(tec, density, pareto_front)