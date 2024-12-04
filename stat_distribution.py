import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter


class ExcelParser:
    def __init__(self, file_path):
        self.data = pd.read_excel(file_path)

    def iterate_rows(self):
        for index, row in self.data.iterrows():
            yield row

    def get_columns(self, row, start_col, end_col):
        return row[start_col:end_col + 1]

    def get_mass_composition(self, row):
        ## 'Fe', 'Ni', 'Co', 'Cr', 'V', 'Cu'
        return list(self.get_columns(row, 1, 6))
    
    def get_phase(self, row):
        phase = row[-1]
        return (phase.split('_')[0]).lower()

    def get_alloy(self, row):
        return row[0]

    def get_Fe(self, row):
        return row[1]
    def get_Ni(self, row):
        return row[2]
    def get_Co(self, row):
        return row[3]
    def get_Cr(self, row):
        return row[4]
    def get_V(self, row):
        return row[5]
    def get_Cu(self, row):
        return row[6]


    def get_density(self, row):
        return row[11]

    def get_tec(self, row):
        return row[18]

    def get_curie_temperature(self, row):
        return row[19]    

    def get_bmag(self, row):
        return row[20]
    
    def get_mags(self, row):
        return row[21]
    
    def calc_descriptor(self, row):
        mags = self.get_mags(row)
        Tc = self.get_curie_temperature(row)
        descriptor = mags / Tc
        print(descriptor, mags, Tc, self.get_tec(row))
        return descriptor

if __name__ == "__main__":
    data_file = "Data_base_DFT_Thermal.xlsx"
    parser = ExcelParser(data_file)

    tec = []
    density = []
    curie = []
    mags = []
    Fe, Ni, Co, Cr, V, Cu = [], [], [], [], [], []
    packings = []



    for row in tqdm(parser.iterate_rows()):
        tec.append(parser.get_tec(row))
        density.append(parser.get_density(row))
        curie.append(parser.get_curie_temperature(row))
        mags.append(parser.get_mags(row))
        Fe.append(parser.get_Fe(row))
        Co.append(parser.get_Co(row))
        Ni.append(parser.get_Ni(row))
        Cr.append(parser.get_Cr(row))
        V.append(parser.get_V(row))
        Cu.append(parser.get_Cu(row))
        packings.append(parser.get_phase(row).split("'")[-1])
    
    ## stat
    plt.figure(figsize=(9,3))
    plt.subplot(141)
    plt.hist(tec, bins=30)
    plt.ylabel('Frequency')
    plt.xlabel('TEC')
    plt.subplot(142)
    plt.hist(density, bins=30)
    plt.xlabel('Density')
    plt.subplot(143)
    plt.hist(curie, bins=30)
    plt.xlabel('Curie Temperature')
    plt.subplot(144)
    plt.hist(mags, bins=30)
    plt.xlabel('Magnetostriciton')
    plt.tight_layout()

    plt.savefig('stat.png', dpi=300)

    ###  vs alloy
    alloys_slices = {
        "Fe-Ni": [0,19],
        "Fe-Co": [19,36],
        "Ni-Co": [36,49],
        "Fe-Ni-Co": [49,110],
        "Fe-Co-Cr": [110,203],
        "Fe-Co-Cr-Cu": [203,235],
        "Fe-Ni-Co-Cr": [235,526],
        "Fe-Co-Ni-V": [526,688],
        "Fe-Co-Ni-V": [688,696],
        "Q2-Q5, Fe-Ni-Co-Cr": [696,700],
        "Q6, Fe-Ni-Co-Cr-Cu": [700,701],
        "Q7, Fe-Ni-Co-Cr": [701,702],
        "Q11-Q16, Fe-Ni-Co-Cr-Cu": [702,708],
        "Q17, Fe-Ni-Co-Cr-Cu": [708,709],
        "Q18-Q19, Fe-Ni-Co-Cr-Cu": [709,711],
        "Q20-Q22, Fe-Ni-Co-Cr-Cu": [711,717]
    }
    plt.figure(figsize=(12,6))
    plt.rcParams['font.size'] = 14
    for name, (start, end) in alloys_slices.items():
        if 'Q' in name:
            s='^'
            alpha=0.5
        else:
            s='o'
            alpha=1
        plt.subplot(121)
        #plt.arrow(5, 8200, -3, -200, head_width=0.5, head_length=20, fc='#222222', ec='#222222')

        plt.scatter(tec[start: end], density[start: end], s=16, label=name, marker=s, alpha=alpha)
        plt.xlabel('TEC')
        plt.ylabel('Density')
        plt.xlim(-10,20)
        plt.legend(ncols=1, fontsize=10, frameon=False, loc='lower left')

        plt.subplot(122)
        #plt.arrow(10, 600, -5, 400, head_width=0.5, head_length=20, fc='#222222', ec='#222222')
        plt.scatter(tec[start: end], curie[start: end], s=16, label=name, marker=s, alpha=alpha)
        plt.xlabel('TEC')
        plt.ylabel('Curie Temperature')
        plt.xlim(-10,20)
    plt.tight_layout()
    plt.savefig('TEC_vs_prop_alloy.png', dpi=300)

    ## vs composition
    plt.figure(figsize=(12, 20))
    plt.rcParams['font.size'] = 14
    plt.rcParams['image.cmap'] = 'YlGnBu'

    elements = {'Fe': Fe, 'Ni': Ni, 'Co': Co, 'Cr': Cr, 'V': V, 'Cu': Cu}
    for i, (element, data) in enumerate(elements.items(), start=1):
        plt.subplot(6, 2, 2 * i - 1)
        cbar = plt.colorbar(plt.scatter(tec, density, s=16, c=data))
        cbar.ax.set_ylabel(element)
        plt.arrow(5, 8200, -3, -200, head_width=0.5, head_length=20, fc='#222222', ec='#222222')
        plt.xlabel('TEC')
        plt.ylabel('Density')
        plt.text(0.05, 0.95, element, transform=plt.gca().transAxes, ha='left', va='top', fontsize=18)
        if i == 1:
            plt.title(f'TEC vs Density')

        plt.subplot(6, 2, 2 * i)
        cbar = plt.colorbar(plt.scatter(tec, curie, s=16, c=data))
        cbar.ax.set_ylabel(element)
        plt.arrow(10, 600, -5, 400, head_width=0.5, head_length=20, fc='#222222', ec='#222222')
        plt.xlabel('TEC')
        plt.ylabel('Curie Temperature')
        plt.text(0.05, 0.95, element, transform=plt.gca().transAxes, ha='left', va='top', fontsize=18)
        if i == 1:
            plt.title(f'TEC vs Curie Temperature')

    plt.tight_layout()

    plt.savefig('TEC_vs_prop.png', dpi=300)

    ## packings
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.size'] = 14
    packing = Counter(packings)
    packing_labels = list(packing.keys())
    packing_values = list(packing.values())
    packing_values_array = np.array(packing_values)
    
    plt.pie(packing_values_array, labels=packing_labels, autopct='%1.1f%%', startangle=90)
    plt.savefig('packing_distribution.png', dpi=300)



