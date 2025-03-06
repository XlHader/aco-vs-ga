import os
import tsplib95
import numpy as np
import pygad
import time
import random


def local_search_2opt(route, distance_matrix, max_iter=1):
    """
    Aplica 2-Opt para intentar mejorar la ruta localmente.
    max_iter controla cuántas pasadas hacemos buscando mejoras.
    """
    n = len(route)

    def route_distance(r):
        """
        Calcula la distancia total de la ruta usando la matriz de distancias.
        """
        return np.sum(distance_matrix[r[:-1], r[1:]]) + distance_matrix[r[-1], r[0]]

    best_route = route.copy()
    best_dist = route_distance(best_route)

    for _ in range(max_iter):
        improved = False
        # Buscar dos índices aleatorios para hacer el intercambio 2-Opt
        for i in range(1, n - 2):  # Empieza en 1, no tiene sentido revisar el primer o el último
            for j in range(i + 1, n - 1):  # j debe ser mayor que i+1
                # Evitar intercambiar elementos consecutivos, ya que no cambiaría nada
                if j - i == 1:
                    continue

                # Realizamos el intercambio 2-Opt, que invierte el segmento entre i y j
                new_route = np.concatenate(
                    (best_route[:i], best_route[i:j+1][::-1], best_route[j+1:]))
                new_dist = route_distance(new_route)

                # Si encontramos una mejor solución, actualizamos
                if new_dist < best_dist:
                    best_dist = new_dist
                    best_route = new_route
                    improved = True

        # Si no mejoramos en esta iteración, terminamos
        if not improved:
            break

    return best_route


def adaptive_mutation_2opt(offspring, ga_instance):
    """
    Mutación adaptativa: la probabilidad de mutar decrece con las generaciones.
    Además, aplicamos 2-Opt con cierta probabilidad (two_opt_prob) para acelerar.
    """
    distance_matrix = ga_instance.user_data["distance_matrix"]
    generations_completed = ga_instance.generations_completed
    max_generations = ga_instance.num_generations

    initial_prob = 0.4
    final_prob = 0.1
    factor = generations_completed / float(max_generations)
    adaptive_prob = initial_prob - factor * (initial_prob - final_prob)

    # Obtén la probabilidad de 2-Opt definida por el usuario (si existe)
    # y si no, usa 0.25 por defecto.
    two_opt_prob = ga_instance.user_data.get("two_opt_prob", 0.25)
    # Número de iteraciones 2-Opt por individuo (puede configurarse)
    two_opt_max_iter = ga_instance.user_data.get("two_opt_max_iter", 1)

    for i in range(len(offspring)):
        # Con prob adaptativa, intercambiamos 2 genes
        if random.random() < adaptive_prob:
            idx1, idx2 = random.sample(range(len(offspring[i])), 2)
            offspring[i][idx1], offspring[i][idx2] = offspring[i][idx2], offspring[i][idx1]

        # Aplicar 2-Opt solo con cierta probabilidad (para reducir costo)
        if random.random() < two_opt_prob:
            route = offspring[i]
            improved_route = local_search_2opt(
                route, distance_matrix, max_iter=two_opt_max_iter)
            offspring[i] = improved_route

    return offspring


def ordered_crossover(parents, offspring_size, ga_instance):
    """
    Implementa el Order Crossover (OX1) para permutaciones en TSP.
    """
    offspring = []
    num_parents = parents.shape[0]
    gene_length = ga_instance.num_genes

    for k in range(offspring_size[0]):
        parent1_idx = k % num_parents
        parent2_idx = (k + 1) % num_parents
        parent1 = parents[parent1_idx]
        parent2 = parents[parent2_idx]

        cp1, cp2 = sorted(random.sample(range(gene_length), 2))
        child = [None] * gene_length

        # Copiar segmento intermedio de parent1
        child[cp1:cp2] = parent1[cp1:cp2].tolist()

        # Rellenar con los genes de parent2 en el orden en que aparecen (sin duplicarlos)
        current_index = cp2
        for gene in parent2:
            if gene not in child:
                if current_index >= gene_length:
                    current_index = 0
                while child[current_index] is not None:
                    current_index += 1
                    if current_index >= gene_length:
                        current_index = 0
                child[current_index] = gene
                current_index += 1

        offspring.append(child)

    return np.array(offspring, dtype=int)


class GAExperiment:
    def __init__(self,
                 instance_file,
                 num_generations=200,
                 sol_per_pop=50,
                 num_parents_mating=10,
                 mutation_probability=0.1,
                 num_threads=1,
                 keep_elitism=2,
                 parent_selection_type="tournament",
                 crossover_type="scattered",
                 crossover_probability=0.9,
                 mutation_type="swap",
                 mutation_percent_genes=15,
                 keep_parents=0,
                 stop_criteria=["saturate_150"],
                 random_seed=42,
                 k_tournament=3,
                 two_opt_prob=0.25,      # prob. de aplicar 2-Opt a cada hijo
                 two_opt_max_iter=1):   # # de iteraciones 2-Opt (por hijo)
        """
        Inicializa los parámetros del algoritmo genético.
        'two_opt_prob' controla con qué prob. se aplica 2-Opt tras la mutación.
        'two_opt_max_iter' cuántas iteraciones de 2-Opt se hacen por individuo.
        """
        self.instance_file = instance_file
        self.num_generations = num_generations
        self.sol_per_pop = sol_per_pop
        self.num_parents_mating = num_parents_mating
        self.mutation_probability = mutation_probability
        self.num_threads = num_threads
        self.keep_elitism = keep_elitism
        self.parent_selection_type = parent_selection_type

        # Elegir crossover
        if crossover_type == "order":
            self.crossover_type = ordered_crossover
        else:
            self.crossover_type = crossover_type

        self.crossover_probability = crossover_probability

        # Elegir mutación
        if mutation_type == "adaptive2opt":
            self.mutation_type = adaptive_mutation_2opt
        elif mutation_type == "adaptive":
            # Tu mutación adaptativa sin 2-Opt (si la tienes en src.ga)
            from src.ga import adaptive_mutation
            self.mutation_type = adaptive_mutation
        else:
            self.mutation_type = mutation_type

        self.mutation_percent_genes = mutation_percent_genes
        self.keep_parents = keep_parents
        self.stop_criteria = stop_criteria
        self.random_seed = random_seed
        self.k_tournament = k_tournament

        # Control de 2-Opt dentro de la mutación
        self.two_opt_prob = two_opt_prob
        self.two_opt_max_iter = two_opt_max_iter

        self.distance_matrix = None

    def run(self):
        """
        Ejecuta el algoritmo genético sobre el problema TSP.
        """
        # Cargar el TSP
        problem = tsplib95.load(self.instance_file)
        graph = problem.get_graph()
        nodes = sorted(list(graph.nodes()))
        n = len(nodes)

        # Crear matriz de distancias
        distance_matrix = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    distance_matrix[i][j] = 0.0
                else:
                    try:
                        distance_matrix[i][j] = graph[nodes[i]
                                                      ][nodes[j]]['weight']
                    except KeyError:
                        distance_matrix[i][j] = 1e9

        self.distance_matrix = distance_matrix

        # Función de aptitud (usa array indexing en lugar de .astype(int) repetidas veces)
        def fitness_func(ga_instance, solution, solution_idx):
            route = solution
            dist = 0
            for k in range(n - 1):
                dist += distance_matrix[route[k], route[k+1]]
            dist += distance_matrix[route[-1], route[0]]
            if dist <= 0:
                return 0
            return 1.0 / dist

        # Población inicial: puras permutaciones
        def init_population_func(sol_per_pop, num_genes):
            return np.array([np.random.permutation(num_genes)
                             for _ in range(sol_per_pop)], dtype=int)

        ga_convergence = []

        # Callback global on_generation (imprime cada 50 gens)
        def on_generation(ga_instance):
            best_fitness = np.max(ga_instance.last_generation_fitness)
            ga_convergence.append(best_fitness)
            # Imprimir cada 50 generaciones en vez de cada 10
            if ga_instance.generations_completed % 10 == 0 or ga_instance.generations_completed == 1:
                print(f"[GA] {os.path.basename(self.instance_file)} "
                      f"- Gen {ga_instance.generations_completed} "
                      f"- Best Fitness: {best_fitness:.6f}")

        # Crear instancia de GA
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
            allow_duplicate_genes=True
        )

        # user_data para accesos en mutación/crossover
        ga_instance.user_data = {
            "distance_matrix": distance_matrix,
            "two_opt_prob": self.two_opt_prob,
            "two_opt_max_iter": self.two_opt_max_iter
        }

        start_time = time.time()
        ga_instance.run()
        elapsed_time = time.time() - start_time

        # Extraer mejor solución
        best_solution, best_solution_fitness, _ = ga_instance.best_solution()
        best_tour = best_solution
        best_length = 0
        for k in range(n - 1):
            best_length += distance_matrix[best_tour[k], best_tour[k+1]]
        best_length += distance_matrix[best_tour[-1], best_tour[0]]

        return best_tour, best_length, ga_convergence, elapsed_time
