import numpy as np
import argparse
import logging
from target import target

class Particle:
    def __init__(self, elements, velocity_scale=0.1, initial_position=None):
        self.elements = elements
        if initial_position is not None:
            # If initial position is provided, handle padding/truncating
            if len(initial_position) < len(elements):
                initial_position += [0] * (len(elements) - len(initial_position))  # Pad with zeros
            elif len(initial_position) > len(elements):
                initial_position = initial_position[:len(elements)]  # Truncate
            self.position = np.array(initial_position)
        else:
            # Otherwise, initialize randomly with a Dirichlet distribution
            self.position = np.random.dirichlet(np.ones(len(elements)))
        self.velocity = np.random.uniform(-velocity_scale, velocity_scale, size=len(elements))
        self.best_position = self.position.copy()
        self.best_score = float("inf")

    def update_velocity(self, global_best_position, inertia, cognitive, social):
        r1, r2 = np.random.rand(len(self.elements)), np.random.rand(len(self.elements))
        cognitive_component = cognitive * r1 * (self.best_position - self.position)
        social_component = social * r2 * (global_best_position - self.position)
        self.velocity = inertia * self.velocity + cognitive_component + social_component
        logging.info(f"Updated velocity: {self.velocity}")

    def update_position(self):
        self.position += self.velocity
        self.position = np.clip(self.position, 0, 1)
        self.position /= np.sum(self.position)  # Normalize
        logging.info(f"Updated position: {self.position}")


class ParticleSwarmOptimization:
    def __init__(self, elements, n_particles=20, generations=100, inertia=0.5, cognitive=2.0, social=2.0, initial_guesses=None):
        self.elements = elements
        self.generations = generations
        self.initial_guesses = initial_guesses  # Store initial guesses
        if initial_guesses is not None:
            logging.info(f"Initial guesses: {initial_guesses}")
            self.n_particles = len(initial_guesses)
            self.particles = self._initialize_particles_from_guesses(initial_guesses)
            self.global_best_position = self.particles[0].position.copy()  # Initialize global best from the first particle
        else:
            logging.info(f"Initializing {n_particles} particles")
            self.n_particles = n_particles
            self.particles = self._initialize_particles(n_particles)
            self.global_best_position = self.particles[0].position.copy()  # Initialize global best from the first particle
        self.global_best_score = float("inf")
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def _initialize_particles(self, n_particles):
        """Randomly initialize particles."""
        particles = []
        for i in range(n_particles):
            particles.append(Particle(self.elements))
        return particles

    def _initialize_particles_from_guesses(self, initial_guesses):
        """Initialize particles using provided initial guesses."""
        particles = []
        for guess in initial_guesses:
            particles.append(Particle(self.elements, initial_position=guess))
        return particles

    def evaluate_fitness(self, particle, generation, **kwargs):
        a, b, c, d = kwargs.get("a", 0.9), kwargs.get("b", 0.1), kwargs.get("c", 0.9), kwargs.get("d", 0.1)
        try:
            score = target(
                elements=self.elements,
                compositions=particle.position,
                generation=generation,
                finalize=None,
                get_density_mode="relax",
                a=a, b=b, c=c, d=d
            )
            if score < particle.best_score:
                particle.best_score = score
                particle.best_position = particle.position.copy()
                logging.info(f"Particle {id(particle)}: New personal best score: {score}")
            if score < self.global_best_score:
                self.global_best_score = score
                self.global_best_position = particle.position.copy()
                logging.info(f"Global best updated: Score: {score}, Position: {self.global_best_position}")
            return score
        except Exception as e:
            logging.error(f"Error evaluating fitness for particle {id(particle)}: {e}")
            return float("inf")  # Handle failed evaluations

    def optimize(self, **kwargs):
        for generation in range(self.generations):
            logging.info(f"Generation {generation} start.")
            for i, particle in enumerate(self.particles, start=1):
                logging.info(f"Particle {i} start.")
                _generation = None if i < len(self.particles) else generation
                self.evaluate_fitness(particle, _generation, **kwargs)
            for particle in self.particles:
                logging.info(f"Particle {i} update.")
                particle.update_velocity(self.global_best_position, self.inertia, self.cognitive, self.social)
                particle.update_position()

            logging.info(f"Generation {generation} - Global Best Score: {self.global_best_score} - Position: {self.global_best_position}")

        return self.global_best_position, self.global_best_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Particle Swarm Optimization for Element Optimization")
    parser.add_argument(
        "--elements",
        type=lambda x: x.split(","),
        default="Fe,Co,Ni,Cr,V,Cu",
        required=False,
        help="Comma-separated list of elements (default: Fe,Co,Ni,Cr,V,Cu)",
    )
    parser.add_argument("--n_particles", type=int, default=20, help="Number of particles (default: 20)")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations (default: 100)")
    parser.add_argument("--inertia", type=float, default=0.5, help="Inertia coefficient (default: 0.5)")
    parser.add_argument("--cognitive", type=float, default=2.0, help="Cognitive coefficient (default: 2.0)")
    parser.add_argument("--social", type=float, default=2.0, help="Social coefficient (default: 2.0)")
    parser.add_argument("--output", type=str, default="pso_debug.log", help="Log filename (default: pso_debug.log)")
    parser.add_argument("--init_mode", type=str, default="init", help="Choose between 'init' and 'random'")
    parser.add_argument("--initial_guesses", type=float, nargs="+", action="append", help="Initial guesses for particle positions. Each guess should be a list of floats, one per element.")
    args = parser.parse_args()
    print(args.init_mode, args.init_mode == "init")
    # Configure logging
    logging.basicConfig(
        filename=args.output,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("----Starting Particle Swarm Optimization----")

    elements = args.elements
    logging.info(f"Elements: {elements}")
    initial_guesses = None
    if args.init_mode == "init":
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

    # Initialize PSO with initial guesses
    pso = ParticleSwarmOptimization(
        elements=elements,
        n_particles=args.n_particles,
        generations=args.generations,
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
        initial_guesses=initial_guesses,  # Pass initial guesses here
    )
    best_position, best_score = pso.optimize(a=0.9, b=0.1, c=0.9, d=0.1)

    logging.info(f"Best Composition: {best_position}, Best Score: {best_score}")
    print("Best Composition:", best_position)
    print("Best Score:", best_score)