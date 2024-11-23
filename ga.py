from target import target
import numpy as np
import logging


logging.basicConfig(filename='ga.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GeneticAlgorithm:
    def __init__(self, elements, population_size=10, generations=100, crossover_rate=0.8, mutation_rate=0.1, init_population=None):
        self.elements = elements
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        if init_population:
            logging.info("Initial population provided, input population size is invalid.")
            self.population = init_population
            self.population_size = len(init_population)
        else:
            logging.info("Initial population not provided, using population size to randomize populations.")
            self.population_size = population_size
            self.population = self.initialize_population()

    def initialize_population(self):
        logging.info("Initializing population.")
        return [self.random_composition() for _ in range(self.population_size)]

    def random_composition(self):
        logging.info("Generating random composition.")
        comp = np.random.dirichlet(np.ones(len(self.elements)), size=1)[0]
        return comp

    def evaluate_fitness(self, comp):        
        logging.info(f"Evaluating fitness for composition: {comp}")
        return target(self.elements, comp)

    def select_parents(self):
        logging.info("Selecting parents.")
        fitness_scores = np.array([self.evaluate_fitness(comp) for comp in self.population])        

        if fitness_scores.size != self.population_size:
            raise ValueError(f"Fitness scores {fitness_scores.size} size does not match population size {self.population_size}.")
        
        total_fitness = np.sum(fitness_scores)
        
        if total_fitness == 0:
            probabilities = np.ones(self.population_size) / self.population_size
        else:
            probabilities = fitness_scores / total_fitness
        
        probabilities = np.clip(probabilities, a_min=0, a_max=1)
        probabilities = probabilities / np.sum(probabilities)
        
        indices = np.arange(self.population_size)        
        selected_indices = np.random.choice(indices, 
                                        size=self.population_size, 
                                        p=probabilities)
        return [self.population[i] for i in selected_indices]

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
        return individual


    def evolve(self):
        logging.info("Evolving.")
        for generation in range(self.generations):
            selected_population = self.select_parents()
            if len(selected_population) % 2 != 0:
                selected_population.pop()
            
            new_population = []
            for i in range(0, len(selected_population), 2):
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                new_population.append(self.mutate(offspring1))
                new_population.append(self.mutate(offspring2))

            self.population = new_population
            best_individual = max(self.population, key=self.evaluate_fitness)
            best_score = self.evaluate_fitness(best_individual)
            logging.info("Generation %d - Best Score: %f", generation, best_score)
            logging.info("Best Individual: %s", best_individual)

        return best_individual, best_score

if __name__ == "__main__":
    elements = ['Fe', 'Ni', 'Co', 'Cr', 'V', 'Cu', 'Ti', 'Al']
    ga = GeneticAlgorithm(elements)
    best_individual, best_score = ga.evolve()
    logging.info(best_individual, best_score)
    target_value = target(elements, comp)
    print("Best Composition:", comp)
    print("Target Value:", target_value)
