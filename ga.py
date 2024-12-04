import argparse
import logging
import numpy as np
from target import target
from constraints_utils import apply_constraints, parse_constraints, mass_to_molar, molar_to_mass

class GeneticAlgorithm:
    def __init__(self, elements, population_size=10, generations=100, crossover_rate=0.8, mutation_rate=0.1, 
                 selection_mode="roulette", init_population=None, constraints={}, a=0.9, b=0.1, c=0.9, d=0.1):
        self.elements = elements
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.selection_mode = selection_mode
        self.constraints = constraints 
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        if init_population:
            logging.info("Initial population provided, manipulating sizes if necessary.")
            self.population = self.manipulate_population_size(init_population)
            self.population_size = len(self.population)
        else:
            logging.info("Initial population not provided, using population size to randomize populations.")
            self.population_size = population_size
            self.population = self.initialize_population()

        logging.info(f"Population size: {self.population_size}")

    def manipulate_population_size(self, population):
        manipulated_population = []
        for individual in population:
            if len(individual) < len(self.elements):
                individual = np.pad(individual, (0, len(self.elements) - len(individual)), mode='constant')
                logging.info(f"Padded individual: {individual}")
            elif len(individual) > len(self.elements):
                individual = individual[:len(self.elements)]
                logging.info(f"Truncated individual: {individual}")
            manipulated_population.append(individual)
        return manipulated_population

    def initialize_population(self):
        logging.info("Initializing population.")
        population = [self.random_composition() for _ in range(self.population_size)]
        if not population:
            raise ValueError("Population initialization failed: population is empty.")
        return population

    def random_composition(self):
        logging.info("Generating random composition.")
        comp = np.random.dirichlet(np.ones(len(self.elements)), size=1)[0]
        return comp

    def evaluate_fitness(self, comp, generation=None):        
        logging.info(f"Evaluating fitness for composition: {comp}")
        if self.constraints:
            molar_comp = mass_to_molar(comp, self.elements)
            comp = apply_constraints(comp, self.elements, self.constraints)
            comp = molar_to_mass(molar_comp, self.elements)
        return target(self.elements, comp, generation=generation, a=self.a, b=self.b, c=self.c, d=self.d, get_density_mode="relax")

    def select_parents(self):
        logging.info("Selecting parents using mode: %s", self.selection_mode)
        if self.selection_mode == "roulette":
            return self.roulette_selection()
        elif self.selection_mode == "tournament":
            return self.tournament_selection()
        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

    def roulette_selection(self):
        fitness_scores = np.array([self.evaluate_fitness(comp) for comp in self.population])
        logging.info(f"Fitness scores: {fitness_scores}")
        total_fitness = np.sum(fitness_scores)
        
        if total_fitness == 0:
            probabilities = np.ones(self.population_size) / self.population_size
        else:
            probabilities = fitness_scores / total_fitness
        
        probabilities = np.clip(probabilities, a_min=0, a_max=1)
        probabilities = probabilities / np.sum(probabilities)
        logging.info(f"Selection probabilities: {probabilities}")
        indices = np.arange(self.population_size)
        selected_indices = np.random.choice(indices, size=self.population_size, p=probabilities)
        parents = [self.population[i] for i in selected_indices]
        return parents

    def tournament_selection(self, tournament_size=6):
        selected_population = []
        for _ in range(self.population_size):
            indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament = [self.population[i] for i in indices]
            best_individual = max(tournament, key=self.evaluate_fitness)
            selected_population.append(best_individual)
        return selected_population

    def crossover(self, parent1, parent2):
        logging.info("Crossover.")
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, len(self.elements) - 1)
            offspring1 = np.concatenate((parent1[:point], parent2[point:]))
            offspring2 = np.concatenate((parent2[:point], parent1[point:]))
            offspring1 /= np.sum(offspring1) 
            offspring2 /= np.sum(offspring2)
            return offspring1, offspring2
        return parent1, parent2

    def mutate(self, individual):
        logging.info("Mutating.")
        if np.random.rand() < self.mutation_rate:
            for _ in range(np.random.randint(1, len(self.elements) // 2 + 1)):  
                point = np.random.randint(len(self.elements))
                individual[point] += np.random.uniform(0.01, 0.05)
                individual = np.clip(individual, a_min=0, a_max=1)
                individual /= np.sum(individual)  
        individual = np.clip(individual, a_min=0, a_max=1)
        return individual

    def evolve(self):
        logging.info("Evolving.")
        for generation in range(self.generations):
            logging.info(f"Generation {generation}")
            selected_population = self.select_parents()
            if len(selected_population) % 2 != 0:
                selected_population.pop()
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(offspring1))
                new_population.append(self.mutate(offspring2))
            if not new_population:
                raise ValueError("Evolution failed: new_population is empty.")
            self.population = new_population

            best_individual = max(self.population, key=self.evaluate_fitness)
            best_score = self.evaluate_fitness(best_individual, generation)
            logging.info("Generation %d - Best Score: %f - Best Individual: %s", generation, best_score, best_individual)
        return best_individual, best_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Element Optimization")
    parser.add_argument("-o", "--output", type=str, default="ga_debug.log", help="Log filename (default: ga.log)")
    parser.add_argument("-e", "--elements", type=str, default="Fe,Ni,Co,Cr,V,Cu,Al,Ti", 
                        help="Comma-separated list of elements (default: predefined list)")
    parser.add_argument("-m", "--init_mode", type=str, default="random", 
                        help="Choose between 'random' and 'init_population'")
    parser.add_argument("-p", "--population_size", type=int, default=10, help="Population size (default: 10)")
    parser.add_argument("-s", "--selection_mode", type=str, default="roulette", 
                        help="Selection mode: 'roulette' or 'tournament' (default: 'roulette')")
    parser.add_argument("-i", "--init_population", type=str, default=None, help="Initial population (default: None)")
    parser.add_argument("--constraints", type=str, default=None, help="Element-wise constraints (e.g., 'Fe<0.5, Al<0.1')")
    
    # Arguments for a, b, c, d
    parser.add_argument("--a", type=float, default=0.9, help="Weight for TEC mean (default: 0.9)")
    parser.add_argument("--b", type=float, default=0.1, help="Weight for TEC std (default: 0.1)")
    parser.add_argument("--c", type=float, default=0.9, help="Weight for density mean (default: 0.9)")
    parser.add_argument("--d", type=float, default=0.1, help="Weight for density std (default: 0.1)")

    args = parser.parse_args()

    logging.basicConfig(filename=args.output, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    elements = args.elements.split(",")
    logging.info("----Starting----")
    logging.info("Elements: %s", elements)

    constraints = parse_constraints(args.constraints)
    if constraints:
        logging.info("Applying element-wise constraints")
        logging.info(f"Constraints: {constraints}")

    # Must be even
    init_population = [
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
        [0.635, 0.305, 0.06, 0.0, 0.0, 0.0]]

    if args.init_mode == "random":
        init_population = None
    ga = GeneticAlgorithm(
        elements=elements,
        population_size=args.population_size,
        generations=100, 
        crossover_rate=0.8, 
        mutation_rate=0.3, 
        selection_mode=args.selection_mode,  
        init_population=init_population,
        constraints=constraints,
        a=args.a, b=args.b, c=args.c, d=args.d)
    best_individual, best_score = ga.evolve()

    logging.info(f"Best Individual: {best_individual}, Best Score: {best_score}")
    print("Best Composition:", best_individual)
    print("Best Score:", best_score)