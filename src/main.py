import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from src import ACOExperiment, GAExperiment


def load_optimal_tour(instance_file):
    """Carga la mejor ruta conocida desde un archivo .opt.tour"""
    opt_tour_file = instance_file.replace(".tsp", ".opt.tour")

    if not os.path.exists(opt_tour_file):
        return None, None

    with open(opt_tour_file, "r") as file:
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


def main():
    parser = argparse.ArgumentParser(
        description="Comparación ACO vs GA para TSPLIB")
    parser.add_argument("--threads", type=int, default=1,
                        help="Número de hilos para paralelizar ACO (default: 1)")
    args = parser.parse_args()

    folder = "TSPLIB"
    if not os.path.exists(folder):
        print(
            f"La carpeta '{folder}' no existe. Coloca los archivos .tsp allí.")
        return

    tsp_files = [os.path.join(folder, f)
                 for f in os.listdir(folder) if f.endswith(".tsp")]
    if not tsp_files:
        print(f"No se encontraron archivos .tsp en la carpeta '{folder}'.")
        return

    summary = []
    for tsp_file in tsp_files:
        instance_name = os.path.basename(tsp_file)
        print(f"\n=== Procesando instancia: {instance_name} ===")

        opt_tour = load_optimal_tour(tsp_file)

        aco_exp = ACOExperiment(instance_file=tsp_file, num_ants=10, num_iterations=250,
                                alpha=1.0, beta=3.0, evaporation_rate=0.1, deposit_factor=2.0,
                                num_threads=args.threads)
        aco_best_solution, aco_best_length, aco_conv, aco_time = aco_exp.run()

        ga_exp = GAExperiment(instance_file=tsp_file, num_generations=1000, sol_per_pop=500,
                              num_parents_mating=150, mutation_probability=0.05, num_threads=args.threads, keep_elitism=5)
        ga_best_tour, ga_best_length, ga_conv, ga_time = ga_exp.run()

        # Comparación con la solución óptima
        opt_distance = None
        if opt_tour is not None:
            opt_distance = sum(
                aco_exp.distance_matrix[opt_tour[i], opt_tour[i + 1]] for i in range(len(opt_tour) - 1))
            # Cerrar el ciclo
            opt_distance += aco_exp.distance_matrix[opt_tour[-1], opt_tour[0]]

        # Cálculo de error relativo
        aco_error = ((aco_best_length - opt_distance) /
                     opt_distance) * 100 if opt_distance else None
        ga_error = ((ga_best_length - opt_distance) /
                    opt_distance) * 100 if opt_distance else None

        summary.append({
            "Instancia": instance_name,
            "ACO Distancia": aco_best_length,
            "ACO Tiempo (s)": round(aco_time, 2),
            "ACO Error (%)": round(aco_error, 2) if aco_error is not None else "N/A",
            "GA Distancia": ga_best_length,
            "GA Tiempo (s)": round(ga_time, 2),
            "GA Error (%)": round(ga_error, 2) if ga_error is not None else "N/A",
            "Óptima Distancia": round(opt_distance, 2) if opt_distance is not None else "N/A"
        })

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

    df = pd.DataFrame(summary)
    print("\n=== Resumen de Resultados ===")
    print(df)


if __name__ == "__main__":
    main()
