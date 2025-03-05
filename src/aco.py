import os
import tsplib95
import acopy
import time
import concurrent.futures
import numpy as np


class ACOExperiment:
    def __init__(self, instance_file, num_ants=10, num_iterations=100,
                 alpha=1.0, beta=3.0, evaporation_rate=0.1, deposit_factor=2.0,
                 num_threads=1):
        """
        Parámetros:
          - instance_file: ruta al archivo TSP. 
          - num_ants: cantidad de hormigas.
          - num_iterations: iteraciones del algoritmo.
          - alpha, beta: parámetros del ACO.
          - evaporation_rate, deposit_factor: parámetros para actualización de feromonas.
          - num_threads: número de hilos/procesos a utilizar.
          - parallel_type: "thread", "process" o None (para secuencial).
        """
        self.instance_file = instance_file
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.deposit_factor = deposit_factor
        self.num_threads = num_threads
        self.distance_matrix = None

    def run(self):
        # Cargar la instancia y obtener el grafo
        problem = tsplib95.load(self.instance_file)
        graph = problem.get_graph()

        nodes = sorted(list(graph.nodes()))
        n = len(nodes)

        self.distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                try:
                    self.distance_matrix[i][j] = graph[nodes[i]
                                                       ][nodes[j]]['weight']
                except KeyError:
                    self.distance_matrix[i][j] = np.inf

        # Inicializar feromonas en cada arista
        for u, v, data in graph.edges(data=True):
            if 'pheromone' not in data:
                data['pheromone'] = 1.0

        # Crear la colonia de hormigas
        colony = acopy.Colony(self.alpha, self.beta)
        best_solution = None
        best_distance = float('inf')
        convergence = []
        start_time = time.time()

        def ant_tour(args):
            ant, graph = args
            return ant.tour(graph)

        for iteration in range(self.num_iterations):
            ants = colony.get_ants(self.num_ants)
            if self.num_threads > 1:
                executor_class = concurrent.futures.ThreadPoolExecutor
                if executor_class:
                    with executor_class(max_workers=self.num_threads) as executor:
                        solutions = list(executor.map(
                            ant_tour, zip(ants, [graph]*len(ants))))
                else:
                    solutions = [ant.tour(graph) for ant in ants]
            else:
                solutions = [ant.tour(graph) for ant in ants]

            iteration_best = None
            iteration_best_distance = float('inf')
            for solution in solutions:
                distance = solution.cost  # 'cost' almacena la distancia total
                if distance < iteration_best_distance:
                    iteration_best_distance = distance
                    iteration_best = solution
                if distance < best_distance:
                    best_distance = distance
                    best_solution = solution

            # Evaporar feromonas
            for u, v, data in graph.edges(data=True):
                data['pheromone'] *= (1 - self.evaporation_rate)

            # Depositar feromonas en el mejor recorrido de la iteración
            if iteration_best is not None:
                tour_nodes = iteration_best.nodes
                for i in range(len(tour_nodes) - 1):
                    u, v = tour_nodes[i], tour_nodes[i+1]
                    graph[u][v]['pheromone'] += self.deposit_factor / \
                        iteration_best_distance
                    graph[v][u]['pheromone'] += self.deposit_factor / \
                        iteration_best_distance

            convergence.append(best_distance)
            print(f"[ACO] {os.path.basename(self.instance_file)} - Iteración {iteration+1}/{self.num_iterations} – Mejor distancia: {best_distance:.2f}")

        elapsed_time = time.time() - start_time
        return best_solution, best_distance, convergence, elapsed_time
