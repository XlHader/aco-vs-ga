import os
import tsplib95
import numpy as np
import pygad
import time


class GAExperiment:
    def __init__(self, instance_file, num_generations=200, sol_per_pop=50,
                 num_parents_mating=10, mutation_probability=0.1, num_threads=1, keep_elitism=2):
        self.instance_file = instance_file
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.mutation_probability = mutation_probability
        self.num_threads = num_threads
        self.keep_elitism = keep_elitism

    def run(self):
        # Cargar la instancia y obtener el grafo
        problem = tsplib95.load(self.instance_file)
        graph = problem.get_graph()
        # Usamos una lista ordenada de nodos para tener índices consistentes
        nodes = sorted(list(graph.nodes()))
        n = len(nodes)

        # Crear la matriz de distancias (n x n) usando los nodos ordenados
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    distance_matrix[i][j] = graph[nodes[i]][nodes[j]]['weight']
                except KeyError:
                    distance_matrix[i][j] = np.inf

        def fitness_func(ga_instance, solution, solution_idx):
            tour = solution.astype(int)
            total_distance = 0
            for i in range(n - 1):
                total_distance += distance_matrix[tour[i]][tour[i+1]]
            total_distance += distance_matrix[tour[-1]][tour[0]]
            return 1.0 / total_distance if total_distance > 0 else 0

        def init_population_func(sol_per_pop, num_genes):
            population = []
            for _ in range(sol_per_pop):
                perm = np.random.permutation(num_genes)
                population.append(perm)
            return np.array(population)

        ga_convergence = []

        def on_generation(ga_instance):
            best_fitness = np.max(ga_instance.last_generation_fitness)
            ga_convergence.append(best_fitness)
            print(f"[GA] {os.path.basename(self.instance_file)} - Generación {ga_instance.generations_completed} – Mejor fitness: {best_fitness:.6f}")

        ga_instance = pygad.GA(num_generations=self.num_generations,
                               num_parents_mating=self.num_parents_mating,
                               fitness_func=fitness_func,
                               sol_per_pop=self.sol_per_pop,
                               num_genes=n,
                               initial_population=init_population_func(
                                   self.sol_per_pop, n),
                               gene_type=int,
                               parent_selection_type="tournament",
                               crossover_type="two_points",
                               crossover_probability=0.9,
                               mutation_type="swap",
                               mutation_probability=self.mutation_probability,
                               mutation_percent_genes=10,
                               keep_parents=0,
                               keep_elitism=self.keep_elitism,
                               stop_criteria=["saturate_150"],
                               on_generation=on_generation,
                               parallel_processing=[
                                   "thread", self.num_threads],
                               random_seed=42
                               )

        start_time = time.time()
        ga_instance.run()
        elapsed_time = time.time() - start_time

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        best_tour = best_solution.astype(int)
        best_length = 0
        for i in range(n - 1):
            best_length += distance_matrix[best_tour[i]][best_tour[i+1]]
        best_length += distance_matrix[best_tour[-1]][best_tour[0]]
        return best_tour, best_length, ga_convergence, elapsed_time
