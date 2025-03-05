import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

from src import ACOExperiment, GAExperiment


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

        aco_exp = ACOExperiment(instance_file=tsp_file, num_ants=10, num_iterations=100,
                                alpha=1.0, beta=3.0, evaporation_rate=0.1, deposit_factor=2.0,
                                num_threads=args.threads)
        aco_best_solution, aco_best_length, aco_conv, aco_time = aco_exp.run()

        ga_exp = GAExperiment(instance_file=tsp_file, num_generations=1000, sol_per_pop=500,
                              num_parents_mating=150, mutation_probability=0.05, num_threads=args.threads, keep_elitism=5)
        ga_best_tour, ga_best_length, ga_conv, ga_time = ga_exp.run()

        summary.append({
            "Instancia": instance_name,
            "ACO Distancia": aco_best_length,
            "ACO Tiempo (s)": round(aco_time, 2),
            "GA Distancia": ga_best_length,
            "GA Tiempo (s)": round(ga_time, 2)
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
