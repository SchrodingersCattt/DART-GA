import argparse
import logging
import numpy as np
from target import target

class GeneticAlgorithm:
    def __init__(self, elements, population_size=10, generations=100, crossover_rate=0.8, mutation_rate=0.1, init_population=None):
        self.elements = elements
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        if init_population:
            logging.info("Initial population provided, input size is invalid.")
            self.population = init_population
            self.population_size = len(init_population)
        else:
            logging.info("Initial population not provided, using population size to randomize populations.")
            self.population_size = population_size
            self.population = self.initialize_population()
        
        logging.info(f"Population size: {self.population_size}")

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
        return target(self.elements, comp, generation)

    def select_parents(self):
        logging.info("Selecting parents.")
        fitness_scores = np.array([self.evaluate_fitness(comp) for comp in self.population])
        logging.info(f"Fitness scores: {fitness_scores}")
        if fitness_scores.size != self.population_size:
            raise ValueError(f"Fitness scores {fitness_scores.size} size does not match population size {self.population_size}.")
        
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
        if not parents:
            raise ValueError("Parent selection failed: selected_population is empty.")
        return parents


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
                individual[point] += np.random.normal(0, 0.05) 
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
    parser.add_argument("-e", "--elements", type=str, default="Fe,Ni,Co,Cr,V,Cu", 
                        help="Comma-separated list of elements (default: predefined list)")
    parser.add_argument("-m", "--mode", type=str, default="random", 
                        help="Choose between 'random' and 'init_population")
    parser.add_argument("-p", "--population_size", type=int, default=10, help="Population size (default: 10)")
    parser.add_argument("-i", "--init_population", type=str, default=None, help="Initial population (default: None)")
    args = parser.parse_args()


    logging.basicConfig(filename=args.output, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    elements = args.elements.split(",")
    logging.info("----Starting----")
    logging.info("Elements: %s", elements)

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
    if args.mode == "random":
        init_population = None
    ga = GeneticAlgorithm(
        elements=elements,
        population_size=16,
        generations=100, 
        crossover_rate=0.8, 
        mutation_rate=0.5, 
        init_population=init_population)
    best_individual, best_score = ga.evolve()

    logging.info(f"Best Individual: {best_individual}, Best Score: {best_score}")
    print("Best Composition:", best_individual)
    print("Best Score:", best_score)
