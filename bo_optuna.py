import optuna
import numpy as np
import argparse
import logging
import re
from target import target
from constraints_utils import apply_constraints, parse_constraints, mass_to_molar, molar_to_mass

def objective(trial, elements, generation, a, b, c, d, constraints):
    # Suggest compositions
    compositions = [trial.suggest_uniform(f"comp_{element}", 0, 1) for element in elements]
    logging.info(f"Generation {generation}: Trial {trial.number} - Raw Compositions: {compositions}")
    # Normalize compositions
    total_mass = sum(compositions)
    normalized_compositions = [comp / total_mass for comp in compositions]
    molar_compositions = mass_to_molar(normalized_compositions, elements)
    
    if constraints:
        logging.info(f"Generation {generation}: Trial {trial.number} - Applying constraints: {constraints}")
        molar_compositions = apply_constraints(molar_compositions, elements, constraints)

    normalized_compositions = molar_to_mass(molar_compositions, elements)
    
    if constraints:
        logging.info(f"Generation {generation}: Trial {trial.number} - Applying constraints: {constraints}")
        normalized_compositions = apply_constraints(normalized_compositions, elements, constraints)

    logging.info(f"Generation {generation}: Trial {trial.number} - Normalized Compositions: {normalized_compositions}")

    # Evaluate the target
    try:
        score = target(
            elements=elements,
            compositions=normalized_compositions,
            finalize=None,
            get_density_mode="relax",
            generation=generation,
            a=a, b=b, c=c, d=d  # Pass a, b, c, d here
        )
    except Exception as e:
        logging.error(f"Generation {generation}: Trial {trial.number} - Error during evaluation: {e}")
        return float("inf")  # Handle failed evaluations
    
    logging.info(f"Generation {generation}: Trial {trial.number} - Score: {-score}")
    return score  


def run_optimization(elements, n_trials=100, initial_guesses=None, a=0.9, b=0.1, c=0.9, d=0.1, constraints={}):
    # Initialize Optuna study
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    generation = 0  # Starting generation

    # Enqueue initial guesses if provided
    if initial_guesses:
        logging.info("Enqueueing initial guesses.")
        for guess in initial_guesses:
            # Adjust initial guesses to match the length of elements
            if len(guess) < len(elements):
                guess += [0] * (len(elements) - len(guess))  # Pad with zeros
            elif len(guess) > len(elements):
                guess = guess[:len(elements)]  # Truncate to match

            normalized_guess = [comp / sum(guess) for comp in guess]
            study.enqueue_trial({f"comp_{el}": normalized_guess[i] for i, el in enumerate(elements)})

    # Optimization process
    def trial_callback(study, trial):
        nonlocal generation
        if generation == 0:
            generation += 1  # Start counting generations after the first trial
        else:
            generation += 1  # Increment generation after each trial

    study.optimize(
        lambda trial: objective(trial, elements, generation, a, b, c, d, constraints),  # Pass constraints here
        n_trials=n_trials,
        callbacks=[trial_callback]
    )
    
    # Logging best parameters
    logging.info("Optimization completed. Best parameters:")
    best_params = study.best_params
    for element in elements:
        logging.info(f"{element}: {best_params[f'comp_{element}']}")

    # Calculate and log normalized best compositions
    best_compositions = [best_params[f"comp_{element}"] for element in elements]
    total_mass = sum(best_compositions)
    normalized_compositions = [comp / total_mass for comp in best_compositions]
    logging.info("Normalized best compositions:")
    for element, comp in zip(elements, normalized_compositions):
        logging.info(f"{element}: {comp:.6f}")

    return normalized_compositions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize target function using Optuna.")
    parser.add_argument(
        "--elements", 
        type=lambda x: x.split(','), 
        default="Fe,Co,Ni,Cr,V,Cu", 
        required=False, 
        help="List of element symbols (comma-separated if default)."
    )
    parser.add_argument("--init_mode", type=str, default="random", 
                        help="Choose between 'init' and 'random'")
    parser.add_argument("--initial_guesses", type=float, nargs="+", action="append", 
                        help="Initial guesses for compositions. Each guess should be a list of floats, one per element.")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of optimization trials.")
    parser.add_argument("-o", "--output", type=str, default="bo_debug.log", help="Log filename (default: bo_debug.log)")

    # Add arguments for a, b, c, d
    parser.add_argument("--a", type=float, default=0.9, help="Weight for TEC mean (default: 0.9)")
    parser.add_argument("--b", type=float, default=0.1, help="Weight for TEC std (default: 0.1)")
    parser.add_argument("--c", type=float, default=0.9, help="Weight for density mean (default: 0.9)")
    parser.add_argument("--d", type=float, default=0.1, help="Weight for density std (default: 0.1)")

    # Add a new argument for constraints
    parser.add_argument("--constraints", type=str, default=None, help="Element-wise constraints (e.g., 'Fe<0.5, Al<0.1')")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        filename=args.output, 
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("----Starting Bayesian Optimization----")

    # Parse arguments
    elements = args.elements

    constraints = parse_constraints(args.constraints)
    if constraints:
        logging.info("Applying element-wise constraints")
        logging.info(f"Constraints: {constraints}")

    if args.init_mode == 'random':
        initial_guesses = None
    elif args.init_mode == 'init':
        logging.info("Initial guesses:")
        #initial_guesses = args.initial_guesses
        initial_guesses =  [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
            [0.95, 0.05, 0.0, 0.0, 0.0, 0.0], 
            [0.9, 0.1, 0.0, 0.0, 0.0, 0.0], 
            [0.85, 0.15, 0.0, 0.0, 0.0, 0.0], 
            [0.91, 0.0, 0.09, 0.0, 0.0, 0.0], 
            [0.801, 0.1, 0.099, 0.0, 0.0, 0.0], 
            [0.37, 0.0, 0.54, 0.09, 0.0, 0.0], 
            [0.37, 0.0, 0.535, 0.095, 0.0, 0.0], 
            [0.375, 0.0, 0.53, 0.095, 0.0, 0.0], 
            [0.365, 0.0, 0.535, 0.1, 0.0, 0.0], 
            [0.636, 0.286, 0.064, 0.014, 0.0, 0.0], 
            [0.621, 0.286, 0.079, 0.014, 0.0, 0.0], 
            [0.485, 0.2, 0.225, 0.0, 0.09, 0.0], 
            [0.605, 0.3, 0.075, 0.0, 0.02, 0.0], 
            [0.635, 0.3, 0.025, 0.0, 0.04, 0.0], 
            [0.635, 0.305, 0.06, 0.0, 0.0, 0.0]
        ]

        for guess in initial_guesses:
            logging.info(f"{guess}")
    
    n_trials = args.n_trials
    logging.info(f"Elements: {elements}")
    a, b, c, d = args.a, args.b, args.c, args.d

    # Run optimization
    best_compositions = run_optimization(elements, n_trials=n_trials, initial_guesses=initial_guesses, a=a, b=b, c=c, d=d, constraints=constraints)
    logging.info("Best Composition: %s", best_compositions)
    print("Best Composition:", best_compositions)
