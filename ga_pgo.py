import pygmo as pg
import numpy as np
import logging

# Setup logging
logging.basicConfig(filename="ga_pgo.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Import your target function
from target import target

class GeneticProblem:
    def __init__(self, elements, penalty_factor=1000):
        self.elements = elements
        self.n_var = len(elements)  # Number of elements in composition
        self.penalty_factor = penalty_factor  # Penalty factor for constraint violation

    # Fitness function for PyGMO
    def fitness(self, x):
        logging.info(f"Evaluating fitness for: {x}")
        try:
            # Objective function value
            fitness_value = target(self.elements, x.tolist())
            
            # Constraint violation (sum(x) must equal 1)
            constraint_violation = abs(np.sum(x) - 1)
            
            # Add penalty for constraint violation
            total_fitness = fitness_value + self.penalty_factor * constraint_violation
            
            logging.info(f"Fitness value: {fitness_value}, Constraint violation: {constraint_violation}, Total fitness: {total_fitness}")
            
            # Return only the penalized objective function
            return [total_fitness]
        except Exception as e:
            logging.error(f"Error during fitness evaluation: {e}")
            raise


    def get_bounds(self):
        return ([0] * self.n_var, [1] * self.n_var)  # Values between 0 and 1

    def get_name(self):
        return "Genetic Problem with Constraint"


def ga():
    # Define the elements
    elements = ['Fe', 'Ni', 'Co', 'Cr', 'V', 'Cu', 'Al', 'Ti']

    # Instantiate the problem
    prob = pg.problem(GeneticProblem(elements))

    # Configure the genetic algorithm
    algo = pg.algorithm(
        pg.nsga2(gen=100)  # Use NSGA-II for handling constraints
    )
    algo.set_verbosity(1)  # Enable detailed step logging

    # Create a population
    pop = pg.population(prob, size=20)  # Initial population of size 20

    # Evolve the population
    for generation in range(10):  # Run for 10 generations
        logging.info(f"Generation {generation}: Best fitness = {pop.get_f()[pop.best_idx()]} Individual: {pop.get_x()[pop.best_idx()]}")
        pop = algo.evolve(pop)

    # Extract the best individual
    best_fitness = pop.get_f()[pop.best_idx()][0]  # Extract the first value (objective function)
    best_individual = pop.get_x()[pop.best_idx()]

    logging.info(f"Final Best Fitness: {best_fitness}")
    logging.info(f"Best Individual: {best_individual}")

    # Print results
    print("Best Composition:", best_individual)
    print("Best Fitness:", best_fitness)


if __name__ == "__main__":
    ga()
