import os
import tsplib95
import numpy as np
import pygad
import time

import numpy as np
import random


def adaptive_mutation(offspring, ga_instance):
    """
    Implementa una mutación adaptativa para TSP que evita duplicados.
    """
    num_offspring = len(offspring)
    generations_completed = ga_instance.generations_completed
    max_generations = ga_instance.num_generations

    # Ajustar la probabilidad de mutación dinámicamente
    initial_prob = 0.4  # Probabilidad inicial alta
    final_prob = 0.1  # Probabilidad final baja
    adaptive_prob = initial_prob - \
        (generations_completed / max_generations) * (initial_prob - final_prob)

    for i in range(num_offspring):
        if np.random.rand() < adaptive_prob:
            # Intercambiar dos ciudades aleatorias en la solución
            idx1, idx2 = np.random.choice(len(offspring[i]), 2, replace=False)
            offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]

    return offspring


def ordered_crossover(parents, offspring_size, ga_instance):
    """
    Implementa el Order Crossover (OX1) para permutaciones en TSP.
    Parámetros:
      - parents: numpy array con los padres.
      - offspring_size: tupla indicando la cantidad de descendientes a generar.
      - ga_instance: instancia de GA, de donde se extrae el número de genes.
    Retorna:
      - offspring: numpy array con los descendientes generados.
    """
    offspring = []
    num_parents = parents.shape[0]
    gene_length = ga_instance.num_genes

    for k in range(offspring_size[0]):
        parent1_idx = k % num_parents
        parent2_idx = (k + 1) % num_parents
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]

        # Seleccionar dos puntos de cruce aleatorios
        cp1, cp2 = sorted(random.sample(range(gene_length), 2))

        # Inicializar el hijo con None
        child = [None] * gene_length

        # Copiar el segmento intermedio de parent1
        child[cp1:cp2] = parent1[cp1:cp2].tolist()

        # Rellenar los espacios vacíos con los genes de parent2 en el orden en que aparecen,
        # omitiendo los que ya se copiaron.
        current_index = cp2
        for gene in parent2:
            if gene not in child:
                if current_index >= gene_length:
                    current_index = 0
                # Buscar la siguiente posición vacía
                while child[current_index] is not None:
                    current_index += 1
                    if current_index >= gene_length:
                        current_index = 0
                child[current_index] = gene
                current_index += 1

        offspring.append(child)

    return np.array(offspring)


class GAExperiment:
    def __init__(self, instance_file, num_generations=200, sol_per_pop=50,
                 num_parents_mating=10, mutation_probability=0.1, num_threads=1,
                 keep_elitism=2, parent_selection_type="tournament",
                 crossover_type="scattered", crossover_probability=0.9,
                 mutation_type="swap", mutation_percent_genes=15,
                 keep_parents=0, stop_criteria=["saturate_150"],
                 random_seed=42, k_tournament=3):
        """
        Inicializa los parámetros del algoritmo genético.
        """
        self.instance_file = instance_file
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.mutation_probability = mutation_probability
        self.num_threads = num_threads
        self.keep_elitism = keep_elitism
        self.parent_selection_type = parent_selection_type
        self.crossover_type = ordered_crossover if crossover_type == "order" else crossover_type
        self.crossover_probability = crossover_probability
        self.mutation_type = adaptive_mutation if mutation_type == "adaptive" else mutation_type
        self.mutation_percent_genes = mutation_percent_genes
        self.keep_parents = keep_parents
        self.stop_criteria = stop_criteria
        self.random_seed = random_seed
        self.k_tournament = k_tournament

    def run(self):
        """
        Ejecuta el algoritmo genético sobre el problema TSP.
        """
        problem = tsplib95.load(self.instance_file)
        graph = problem.get_graph()
        nodes = sorted(list(graph.nodes()))
        n = len(nodes)

        # Crear matriz de distancias
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    distance_matrix[i][j] = graph[nodes[i]][nodes[j]]['weight']
                except KeyError:
                    distance_matrix[i][j] = np.inf

        def fitness_func(_, solution, __):
            tour = solution.astype(int)
            total_distance = sum(
                distance_matrix[tour[i]][tour[i + 1]] for i in range(n - 1))
            total_distance += distance_matrix[tour[-1]][tour[0]]
            return 1.0 / total_distance if total_distance > 0 else 0

        def init_population_func(sol_per_pop, num_genes):
            return np.array([np.random.permutation(num_genes) for _ in range(sol_per_pop)])

        ga_convergence = []

        def on_generation(ga_instance):
            best_fitness = np.max(ga_instance.last_generation_fitness)
            ga_convergence.append(best_fitness)
            print(
                f"[GA] {os.path.basename(self.instance_file)} - Gen {ga_instance.generations_completed} - Best fitness: {best_fitness:.6f}")

        ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            fitness_func=fitness_func,
            sol_per_pop=self.sol_per_pop,
            num_genes=n,
            initial_population=init_population_func(self.sol_per_pop, n),
            gene_type=int,
            parent_selection_type=self.parent_selection_type,
            crossover_type=self.crossover_type,
            crossover_probability=self.crossover_probability,
            mutation_type=self.mutation_type,
            mutation_probability=self.mutation_probability,
            mutation_percent_genes=self.mutation_percent_genes,
            keep_parents=self.keep_parents,
            keep_elitism=self.keep_elitism,
            stop_criteria=self.stop_criteria,
            on_generation=on_generation,
            parallel_processing=["thread", self.num_threads],
            random_seed=self.random_seed,
            K_tournament=self.k_tournament,
        )

        start_time = time.time()
        ga_instance.run()
        elapsed_time = time.time() - start_time

        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        best_tour = best_solution.astype(int)
        best_length = sum(
            distance_matrix[best_tour[i]][best_tour[i + 1]] for i in range(n - 1))
        best_length += distance_matrix[best_tour[-1]][best_tour[0]]

        return best_tour, best_length, ga_convergence, elapsed_time
