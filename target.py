import glob
import logging
import copy
import dpdata
import numpy as np
from tqdm import tqdm
from pymatgen.core.structure import Structure, Element, Lattice
from deepmd.pt.infer.deep_eval import DeepProperty
from calc_density import calculate_density
from deepmd.calculator import DP as DPCalculator

atomic_mass_file = "constant/atomic_mass.json"

def mk_template_supercell(packing: str):
    if "fcc" in packing:
        s = Structure.from_file("struct_template/fcc-Ni_mp-23_conventional_standard.cif")
        return s.make_supercell([5,5,5])
    elif "bcc" in packing:
        s = Structure.from_file("struct_template/bcc-Fe_mp-13_conventional_standard.cif")
        return s.make_supercell([6,6,6])
    elif "hcp" in packing:
        s = Structure.from_file("struct_template/hcp-Co_mp-54_conventional_standard.cif")
        return s.make_supercell([6,6,6])
    else:
        raise ValueError(f"{packing} not supported")

def normalize_composition(composition: list, total: int) -> list:
    if not composition or total <= 0:
        print("Warning: Invalid input. Returning None.")
        return None

    total_composition = sum(composition)
    if total_composition == 0:
        print("Warning: Composition is all zeros. Returning None.")
        return None

    norm_composition_float = [c / total_composition * total for c in composition]    
    norm_composition = [int(round(x)) for x in norm_composition_float]
    
    diff = sum(norm_composition) - total
    if diff != 0:
        max_index = norm_composition.index(max(norm_composition))
        norm_composition[max_index] -= diff
    
    if abs(sum(norm_composition) - total) > 1:
        print(f"Warning: Normalization failed. Sum: {sum(norm_composition)}, Target: {total}")
        print(f"Original: {composition}, Normalized: {norm_composition}")
        return None

    return norm_composition

def mass_to_molar(mass_dict: dict):
    molar_composition = []
    for kk in mass_dict.keys():
        molar_composition.append(mass_dict[kk] / Element(kk).atomic_mass)
    return molar_composition

def comp2struc(element_list, mass_composition, packing):
    MAX = 10
    supercell = mk_template_supercell(packing)
    pmg_elements = [Element(e) for e in element_list]
    composition = mass_to_molar(dict(zip(element_list, mass_composition)))

    atom_num = len(supercell)
    normalized_composition = normalize_composition(copy.deepcopy(composition), atom_num)
    logging.info(f"Normalized composition: {normalized_composition}")

    if normalized_composition is None or sum(normalized_composition) != atom_num:
        raise ValueError("Composition normalization failed.")

    strucutre_list = []
    for rand_seed in range(MAX):
        np.random.seed(rand_seed)
        replace_mapping = zip(pmg_elements, normalized_composition)
        atom_range = np.array(range(atom_num))
        selected_indices = []
        for ii, (element, num) in enumerate(replace_mapping):
            available_indices = np.setdiff1d(atom_range, selected_indices)

            if num < 0 or len(available_indices) < num:
                raise ValueError(f"Invalid atom replacement: num={num}, available={len(available_indices)}")

            chosen_idx = np.random.choice(available_indices, num, replace=False)
            selected_indices.extend(chosen_idx)
            for jj in chosen_idx:
                ss = supercell.replace(jj, element)

        strucutre_list.append(ss)

    return strucutre_list



def change_type_map(origin_type: list, data_type_map, model_type_map):
    final_type = []
    for single_type in origin_type:
        element = data_type_map[single_type]
        final_type.append(np.where(np.array(model_type_map)==element)[0][0])

    return final_type

def z_core(array, mean = None, std = None):
    return (array - mean) / std

def norm2orig(pred, mean, std):
    return pred * std + mean

def pred(model, structure):    
    d = dpdata.System(structure, fmt='pymatgen/structure')
    orig_type_map = d.data["atom_names"]
    coords = d.data['coords']
    cells = d.data['cells']
    atom_types = change_type_map(d.data['atom_types'], orig_type_map, model.get_type_map())

    pred = model.eval(
        coords=coords, 
        atom_types=atom_types, 
        cells=cells
    )[0]

    return pred

def get_packing(elements, compositions): 
    ## TODO 
    packing = 'bcc'
    return packing

def target(
        elements, 
        compositions, 
        a=0.9, b=0.1, c=0.9, d=0.1,
        generation=None, 
        finalize=None, 
        get_density_mode="relax", 
        calculator=DPCalculator("/mnt/data_nas/guomingyu/PROPERTIES_PREDICTION/RELAX_Density_Calculation/alloy.pth")
    ):
    logging.info(f"a: {a}, b: {b}, c: {c}, d: {d}, compositions: {compositions}")
    packing = get_packing(elements, compositions)

    tec_models = glob.glob('models/tec*.pt')
    tec_models = (DeepProperty(model) for model in tec_models)

    struct_list = comp2struc(elements, compositions, packing=packing)

    ## TEC is original data, density is normalized data
    pred_tec = [z_core(pred(m, s), mean=9.76186694677871, std=4.3042156360248125) for m in tec_models for s in tqdm(struct_list)]
    pred_tec_mean = np.mean(pred_tec)
    pred_tec_std = np.std(pred_tec)

    if get_density_mode == "relax":
        assert calculator is not None, "calculator is not provided"
        pred_density = [z_core(calculate_density(s, calculator), mean= 8331.903892865434, std=182.21803336559455) for s in tqdm(struct_list)]
    elif get_density_mode == "predict" or get_density_mode == "pred":
        density_models = glob.glob('models/density*.pt')
        density_models = (DeepProperty(model) for model in density_models)
        pred_density = [pred(m, s) for m in density_models for s in tqdm(struct_list)]
    else:
        raise ValueError(f"{get_density_mode} not supported, choose between relax, predict or pred")
    pred_density_mean = np.mean(pred_density)
    pred_density_std = np.std(pred_density)
    target = a * (-1* pred_tec_mean) + b * pred_tec_std + c * (-1* pred_density_mean) + d * pred_density_std

    if generation is not None:
        logging.info(
            f"""
            ====\n
            - Generation {generation}, 
            - pred_tec_mean: {norm2orig(pred_tec_mean, mean=9.76186694677871, std=4.3042156360248125)},
            - pred_density_mean: {norm2orig(pred_density_mean, mean= 8331.903892865434, std=182.21803336559455)},
            - target: {target}
            ----\n
            """)
    if finalize is not None:
        logging.info(f"Final target: {target}")
        logging.info(
            f"""
            ====\n
            - pred_tec_mean: {norm2orig(pred_tec_mean, mean=9.76186694677871, std=4.3042156360248125)},
            - pred_density_mean: {norm2orig(pred_density_mean, mean= 8331.903892865434, std=182.21803336559455)},
            - pred_tec_std: {np.std([norm2orig(tec, mean=9.76186694677871, std=4.3042156360248125) for tec in pred_density])},
            - pred_density_std: {np.std([norm2orig(den, mean= 8331.903892865434, std=182.21803336559455) for den in pred_density])},
            - target: {target}
            ----\n
            """)


    return target



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    calculator = DPCalculator("/mnt/data_nas/guomingyu/PROPERTIES_PREDICTION/RELAX_Density_Calculation/alloy.pth")
    comp = [
        1.58534604e-01,
        3.81925319e-02,
        9.92886136e-02,
        1.09126615e-02,
        6.92985901e-01,
        8.56879332e-05, 
        0.00000000e+00, 
        0.00000000e+00
    ]
    elements = ['Fe', 'Ni', 'Co', 'Cr', 'V', 'Cu', 'Al', 'Ti']
    ss = comp2struc(elements, comp, 'bcc')
    from time import time
    start = time()
    target = target(elements, comp, finalize=True, get_density_mode="pred", calculator=calculator)
    end = time()
    print(end - start)
    print(target)