import optuna
import numpy as np
import argparse
import logging
from target import target

def objective(trial, elements, generation):
    # Suggest compositions
    compositions = [trial.suggest_uniform(f"comp_{element}", 0, 1) for element in elements]
    logging.info(f"Generation {generation}: Trial {trial.number} - Raw Compositions: {compositions}")
    
    # Normalize compositions
    total_mass = sum(compositions)
    normalized_compositions = [comp / total_mass for comp in compositions]
    logging.info(f"Generation {generation}: Trial {trial.number} - Normalized Compositions: {normalized_compositions}")

    # Evaluate the target
    try:
        score = target(
            elements=elements,
            compositions=normalized_compositions,
            finalize=None,
            get_density_mode="relax",
            generation=generation,
        )
    except Exception as e:
        logging.error(f"Generation {generation}: Trial {trial.number} - Error during evaluation: {e}")
        return float("inf")  # Handle failed evaluations
    
    logging.info(f"Generation {generation}: Trial {trial.number} - Score: {-score}")
    return -score  # Minimize negative score to maximize the original target


def run_optimization(elements, n_trials=100, initial_guesses=None):
    # Initialize Optuna study
    study = optuna.create_study(direction="minimize")
    generation = 0

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
        generation += 1

    study.optimize(
        lambda trial: objective(trial, elements, generation),
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

    # Run optimization
    best_compositions = run_optimization(elements, n_trials=n_trials, initial_guesses=initial_guesses)
    logging.info("Best Composition: %s", best_compositions)
    print("Best Composition:", best_compositions)
