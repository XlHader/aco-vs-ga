import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from src.aco import ACOExperiment
from src.ga import GAExperiment


def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Comparación ACO vs GA para TSP")

    # Archivos de entrada obligatorios
    parser.add_argument("--tsp", type=str, required=True,
                        help="Ruta al archivo .tsp")
    parser.add_argument("--tour", type=str, required=True,
                        help="Ruta al archivo .opt.tour")

    # Parámetro generales
    parser.add_argument("--threads", type=int, default=1,
                        help="Número de hilos para paralelismo")

    # Configuración de ACO
    parser.add_argument("--aco_ants", type=int, default=10,
                        help="Número de hormigas en ACO")
    parser.add_argument("--aco_iterations", type=int,
                        default=250, help="Número de iteraciones en ACO")

    # Configuración de GA
    parser.add_argument("--ga_generations", type=int,
                        default=250, help="Número de generaciones en GA")
    parser.add_argument("--ga_pop_size", type=int, default=500,
                        help="Tamaño de la población en GA")
    parser.add_argument("--ga_parents", type=int,
                        default=150, help="Número de padres en GA")
    parser.add_argument("--ga_mutation", type=str, default="0.05",
                        help="Probabilidad de mutación en GA (Ejemplo: '0.1' o '0.4,0.1' para adaptativo)")
    parser.add_argument("--ga_keep_elite", type=int, default=5,
                        help="Número de mejores individuos retenidos")
    parser.add_argument("--ga_selection", type=str, default="tournament",
                        help="Método de selección de padres en GA ('tournament', 'roulette', etc.)")
    parser.add_argument("--ga_crossover", type=str, default="order",
                        help="Tipo de crossover en GA ('scattered', 'two_points', 'order')")
    parser.add_argument("--ga_crossover_prob", type=float, default=0.9,
                        help="Probabilidad de crossover en GA")
    parser.add_argument("--ga_mutation_type", type=str, default="swap",
                        help="Tipo de mutación en GA ('swap', 'scramble', 'adaptive')")
    parser.add_argument("--ga_mutation_percent", type=int, default=15,
                        help="Porcentaje de genes mutados en GA")
    parser.add_argument("--ga_keep_parents", type=int, default=0,
                        help="Número de padres mantenidos en la siguiente generación")
    parser.add_argument("--ga_stop_criteria", type=str, default="saturate_150",
                        help="Criterio de parada en GA (Ejemplo: 'saturate_150,reach_0.00001')")
    parser.add_argument("--ga_seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad")
    parser.add_argument("--ga_k_tournament", type=int, default=3,
                        help="Número de individuos en el torneo para selección")

    return parser.parse_args()


def validate_files(tsp_file, tour_file):
    """Verifica que los archivos de entrada existan, si no lanza una excepción."""
    if not os.path.exists(tsp_file):
        raise FileNotFoundError(
            f"❌ El archivo '{tsp_file}' no existe. Verifica la ruta.")

    if not os.path.exists(tour_file):
        raise FileNotFoundError(
            f"❌ El archivo '{tour_file}' no existe. Verifica la ruta.")


def load_optimal_tour(tour_file):
    """Carga la mejor ruta conocida desde un archivo .opt.tour"""
    with open(tour_file, "r") as file:
        lines = file.readlines()

    tour = []
    recording = False
    for line in lines:
        if "TOUR_SECTION" in line:
            recording = True
            continue
        if recording:
            if "-1" in line or "EOF" in line:
                break
            tour.append(int(line.strip()) - 1)

    return tour


def run_experiments(tsp_file, tour_file, args):
    """Ejecuta ACO y GA en una instancia de TSP y compara con la solución óptima."""
    instance_name = os.path.basename(tsp_file)
    print(f"\n=== Procesando instancia: {instance_name} ===")

    opt_tour = load_optimal_tour(tour_file)

    if "," in args.ga_mutation:
        args.ga_mutation = [float(m) for m in args.ga_mutation.split(",")]
    else:
        args.ga_mutation = float(args.ga_mutation)

    # Ejecutar ACO
    aco_exp = ACOExperiment(instance_file=tsp_file,
                            num_ants=args.aco_ants,
                            num_iterations=args.aco_iterations,
                            num_threads=args.threads)
    aco_best_solution, aco_best_length, aco_conv, aco_time = aco_exp.run()

    # Ejecutar GA
    ga_exp = GAExperiment(
        instance_file=args.tsp,
        num_generations=args.ga_generations,
        sol_per_pop=args.ga_pop_size,
        num_parents_mating=args.ga_parents,
        mutation_probability=args.ga_mutation,
        num_threads=args.threads,
        keep_elitism=args.ga_keep_elite,
        parent_selection_type=args.ga_selection,
        crossover_type=args.ga_crossover,
        crossover_probability=args.ga_crossover_prob,
        mutation_type=args.ga_mutation_type,
        mutation_percent_genes=args.ga_mutation_percent,
        keep_parents=args.ga_keep_parents,
        stop_criteria=args.ga_stop_criteria.split(","),
        random_seed=args.ga_seed
    )
    ga_best_tour, ga_best_length, ga_conv, ga_time = ga_exp.run()

    # Comparación con la solución óptima
    opt_distance = sum(
        aco_exp.distance_matrix[opt_tour[i], opt_tour[i + 1]] for i in range(len(opt_tour) - 1))
    opt_distance += aco_exp.distance_matrix[opt_tour[-1], opt_tour[0]]

    # Cálculo del error porcentual
    aco_error = ((aco_best_length - opt_distance) / opt_distance) * 100
    ga_error = ((ga_best_length - opt_distance) / opt_distance) * 100

    # Devolver resultados
    return {
        "Instancia": instance_name,
        "ACO Distancia": aco_best_length,
        "ACO Tiempo (s)": round(aco_time, 2),
        "ACO Error (%)": round(aco_error, 2),
        "GA Distancia": ga_best_length,
        "GA Tiempo (s)": round(ga_time, 2),
        "GA Error (%)": round(ga_error, 2),
        "Óptima Distancia": round(opt_distance, 2),
        "ACO Convergencia": aco_conv,
        "GA Convergencia": ga_conv
    }


def plot_convergence(instance_name, aco_conv, ga_conv):
    """Genera gráficos de convergencia para ACO y GA."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(aco_conv, marker='o', linestyle='-')
    plt.title(f"Convergencia ACO - {instance_name}")
    plt.xlabel("Iteración")
    plt.ylabel("Distancia mínima")

    plt.subplot(1, 2, 2)
    plt.plot(ga_conv, marker='o', linestyle='-')
    plt.title(f"Convergencia GA - {instance_name}")
    plt.xlabel("Generación")
    plt.ylabel("Fitness")

    plt.tight_layout()
    plt.show()


def main():
    args = parse_arguments()

    # Validar archivos
    validate_files(args.tsp, args.tour)

    # Ejecutar experimentos y obtener resultados
    result = run_experiments(args.tsp, args.tour, args)

    # Graficar convergencia
    plot_convergence(
        result["Instancia"], result["ACO Convergencia"], result["GA Convergencia"])

    # Mostrar resumen final
    df = pd.DataFrame([result]).drop(
        columns=["ACO Convergencia", "GA Convergencia"])
    print("\n=== Resumen de Resultados ===")
    print(df)


if __name__ == "__main__":
    main()
