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
    parser.add_argument("--ga_mutation", type=float,
                        default=0.05, help="Probabilidad de mutación en GA")
    parser.add_argument("--ga_keep_elite", type=int, default=5,
                        help="Número de mejores individuos retenidos")

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

    # 🔹 Ejecutar ACO
    aco_exp = ACOExperiment(instance_file=tsp_file,
                            num_ants=args.aco_ants,
                            num_iterations=args.aco_iterations,
                            num_threads=args.threads)
    aco_best_solution, aco_best_length, aco_conv, aco_time = aco_exp.run()

    # 🔹 Ejecutar GA
    ga_exp = GAExperiment(instance_file=tsp_file,
                          num_generations=args.ga_generations,
                          sol_per_pop=args.ga_pop_size,
                          num_parents_mating=args.ga_parents,
                          mutation_probability=args.ga_mutation,
                          num_threads=args.threads,
                          keep_elitism=args.ga_keep_elite)
    ga_best_tour, ga_best_length, ga_conv, ga_time = ga_exp.run()

    # Comparación con la solución óptima
    opt_distance = sum(
        aco_exp.distance_matrix[opt_tour[i], opt_tour[i + 1]] for i in range(len(opt_tour) - 1))
    opt_distance += aco_exp.distance_matrix[opt_tour[-1], opt_tour[0]]

    # 🔹 Cálculo del error porcentual
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

    # 🔹 Graficar convergencia
    plot_convergence(
        result["Instancia"], result["ACO Convergencia"], result["GA Convergencia"])

    # Mostrar resumen final
    df = pd.DataFrame([result]).drop(
        columns=["ACO Convergencia", "GA Convergencia"])
    print("\n=== Resumen de Resultados ===")
    print(df)


if __name__ == "__main__":
    main()
