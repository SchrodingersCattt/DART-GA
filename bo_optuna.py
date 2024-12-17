import optuna
import numpy as np
import argparse
import logging
import re
from target import target
from constraints_utils import apply_constraints, parse_constraints, mass_to_molar, molar_to_mass

def objective(trial, elements, generation, a, b, c, d, constraints, step_size=None, get_density_mode="weighted_avg"):
    # Suggest compositions
    compositions = [trial.suggest_uniform(f"comp_{element}", 0, 1) for element in elements]
    logging.info(f"Generation {generation}: Trial {trial.number} - Raw Compositions: {compositions}")
    
    # Apply step size limit based on the generation number
    if generation > 0 and step_size is not None:
        # Restrict the change in composition from the previous generation
        prev_compositions = trial.user_attrs.get('prev_compositions', [0]*len(elements))  # Keep track of previous compositions
        compositions = [
            np.clip(comp, prev_comp - step_size, prev_comp + step_size)
            for comp, prev_comp in zip(compositions, prev_compositions)
        ]
    
    # Normalize compositions
    total_mass = sum(compositions)
    normalized_compositions = [comp / total_mass for comp in compositions]
    molar_compositions = mass_to_molar(normalized_compositions, elements)
    
    if constraints:
        logging.info(f"Generation {generation}: Trial {trial.number} - Applying constraints: {constraints}")
        molar_compositions = apply_constraints(molar_compositions, elements, constraints)

    normalized_compositions = molar_to_mass(molar_compositions, elements)
    logging.info(f"Generation {generation}: Trial {trial.number} - Normalized Compositions: {normalized_compositions}")

    # Evaluate the target
    try:
        score = target(
            elements=elements,
            compositions=normalized_compositions,
            finalize=None,
            get_density_mode=get_density_mode,
            generation=generation,
            a=a, b=b, c=c, d=d  # Pass a, b, c, d here
        )
    except Exception as e:
        import traceback
        logging.error(f"Generation {generation}: Trial {trial.number} - Error during evaluation: {e, traceback.format_exc()}")
        return float("inf")  # Handle failed evaluations
    
    logging.info(f"Generation {generation}: Trial {trial.number} - Score: {-score}")
    
    # Store current compositions for the next trial
    trial.set_user_attr('prev_compositions', compositions)

    return score

def run_optimization(
    elements, n_trials=100, initial_guesses=None, 
    a=0.9, b=0.1, c=0.9, d=0.1, constraints={}, stepsize=None,
    get_density_mode='weighted_avg'):
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
        lambda trial: objective(trial, elements, generation, a, b, c, d, constraints, stepsize, get_density_mode),  # Pass constraints here
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
    parser.add_argument("--stepsize", type=float, default=None, help="Step size for the optimization process")
    parser.add_argument("--get_density_mode", type=str, default="weighted_avg", help="Mode for density calculation (e.g. pred, relax, default: 'weighted_avg').")
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
    get_density_mode = args.get_density_mode
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
    best_compositions = run_optimization(
        elements, n_trials=n_trials, initial_guesses=initial_guesses, 
        a=a, b=b, c=c, d=d, constraints=constraints,
        get_density_mode=get_density_mode)
    logging.info("Best Composition: %s", best_compositions)
