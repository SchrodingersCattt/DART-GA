import logging
import json
import numpy as np

# constraints
def parse_constraints(constraints_str):
    constraints = {}
    if constraints_str:
        for constraint in constraints_str.split(','):
            if '<' in constraint:
                element, condition = constraint.split('<')
                constraints[element] = f"<{condition}"
            elif '>' in constraint:
                element, condition = constraint.split('>')
                constraints[element] = f">{condition}"
            elif '=' in constraint:
                element, condition = constraint.split('=')
                constraints[element] = f"={condition}"
            else:
                logging.warning(f"Invalid constraint format: {constraint}")
    return constraints

def apply_constraints(compositions, elements, constraints):
    original_sum = sum(compositions)
    modified_compositions = compositions.copy()

    # First pass: Apply constraints
    for i, element in enumerate(elements):
        if element in constraints:
            condition_str = constraints[element]
            condition, value = condition_str[0], float(condition_str[1:])

            if condition == '<' and modified_compositions[i] > value:
                modified_compositions[i] = value
            if condition == '>' and modified_compositions[i] < value:
                modified_compositions[i] = value
            if condition == '=' and modified_compositions[i] != value:
                modified_compositions[i] = value

    # Calculate the remaining composition to be distributed
    constrained_sum = sum(modified_compositions[i]
                          for i, element in enumerate(elements)
                          if element in constraints)
    remaining = 1.0 - constrained_sum

    # Distribute the remaining composition proportionally among unconstrained elements
    unconstrained_indices = [i for i, element in enumerate(elements)
                             if element not in constraints]

    if unconstrained_indices:
        original_unconstrained_sum = sum(compositions[i] for i in unconstrained_indices)
        if original_unconstrained_sum > 0:
            ratio = remaining / original_unconstrained_sum
            for i in unconstrained_indices:
                modified_compositions[i] *= ratio

    modified_compositions = np.clip(modified_compositions, 0, 1)  # Ensuring no values are out of bounds
    modified_compositions /= np.sum(modified_compositions)  # Renormalize after modification

    return modified_compositions


## atomic mass and molar mass conversion
atoms_mass_file = "/mnt/data_nas/guomingyu/PROPERTIES_PREDICTION/Genetic_Alloy/constant/atomic_mass.json"
with open(atoms_mass_file, 'r') as atoms_mass_file:
    atomic_mass = json.load(atoms_mass_file)
def mass_to_molar(mass_comp: list, element_list: list) -> list:
    total_mass = sum(mass_comp)
    molar_compositions = [
        (mass_comp[i] / atomic_mass[element_list[i]]) for i in range(len(element_list))
    ]
    
    # Normalize to get molar fractions
    total_moles = sum(molar_compositions)
    molar_fractions = [moles / total_moles for moles in molar_compositions]
    
    return molar_fractions

def molar_to_mass(molar_comp: list, element_list: list) -> list:
    total_moles = sum(molar_comp)
    mass_compositions = [
        molar_comp[i] * atomic_mass[element_list[i]] for i in range(len(element_list))
    ]
    
    # Normalize to get mass fractions
    total_mass = sum(mass_compositions)
    mass_fractions = [mass / total_mass for mass in mass_compositions]
    
    return mass_fractions

def sigmoid(x):
    return 1 / (1 + np.exp(-x))