import numpy as np
import argparse
import logging
from target import target

class Particle:
    def __init__(self, elements, velocity_scale=0.1):
        self.elements = elements
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
    def __init__(self, elements, n_particles=20, generations=100, inertia=0.5, cognitive=2.0, social=2.0):
        self.elements = elements
        self.generations = generations
        self.particles = [Particle(elements) for _ in range(n_particles)]
        self.global_best_position = None
        self.global_best_score = float("inf")
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

    def evaluate_fitness(self, particle, generation):
        try:
            score = target(
                elements=self.elements,
                compositions=particle.position,
                generation=generation,
                finalize=None,
                get_density_mode="pred",
                calculator=None,
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

    def optimize(self):
        for generation in range(self.generations):
            logging.info(f"Generation {generation} start.")
            for particle in self.particles:
                self.evaluate_fitness(particle, generation)
            for particle in self.particles:
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
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        filename=args.output,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("----Starting Particle Swarm Optimization----")

    elements = args.elements
    logging.info(f"Elements: {elements}")

    pso = ParticleSwarmOptimization(
        elements=elements,
        n_particles=args.n_particles,
        generations=args.generations,
        inertia=args.inertia,
        cognitive=args.cognitive,
        social=args.social,
    )
    best_position, best_score = pso.optimize()

    logging.info(f"Best Composition: {best_position}, Best Score: {best_score}")
    print("Best Composition:", best_position)
    print("Best Score:", best_score)
