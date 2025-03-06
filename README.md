# ACO vs GA - Comparación de Algoritmos para TSP

Este proyecto implementa y compara dos algoritmos de optimización para resolver el Problema del Viajero (TSP) usando instancias de TSPLIB:

- **ACO (Optimización de Colonias de Hormigas)**
- **GA (Algoritmos Genéticos)**

Se ha optimizado el rendimiento con paralelización configurable.

## Características
- Implementación modular con clases separadas para ACO y GA.
- Configuración flexible mediante argumentos de línea de comandos.
- Soporte para ejecución en paralelo con `threads`.

## Requisitos

- Python 3.10+
- Dependencias instaladas:
  ```sh
  pip install -r requirements.txt
  ```
- Instancias TSP en la carpeta `TSPLIB/`

## Uso

### Ejecución Manual

## Berlín 52

```sh
python3 -m src.main \
  --tsp TSPLIB/berlin52.tsp --tour TSPLIB/berlin52.opt.tour \
  --aco_ants 20 --aco_iterations 300 --aco_max_stagnation 50 \
  --threads 16 \
  --ga_generations 500 \
  --ga_pop_size 200 \
  --ga_parents 50 \
  --ga_mutation 0.1 \
  --ga_keep_elite 5 \
  --ga_selection tournament \
  --ga_crossover order \
  --ga_crossover_prob 0.9 \
  --ga_mutation_type adaptive2opt \
  --ga_mutation_percent 15 \
  --ga_keep_parents 0 \
  --ga_stop_criteria saturate_20 \
  --ga_seed 42 \
  --ga_k_tournament 3
```

## ch130 

```sh
python3 -m src.main \
    --tsp TSPLIB/ch130.tsp \
    --tour TSPLIB/ch130.opt.tour \
    --threads 16 \
    \
    --aco_ants 25 \
    --aco_iterations 500 \
    --aco_max_stagnation 100 \
    --aco_alpha 1.0 \
    --aco_beta 5.0 \
    --aco_evaporation_rate 0.2 \
    --aco_deposit_factor 3.0 \
    \
    --ga_generations 1200 \
    --ga_pop_size 250 \
    --ga_parents 80 \
    --ga_mutation 0.1 \
    --ga_keep_elite 5 \
    --ga_selection tournament \
    --ga_crossover order \
    --ga_crossover_prob 0.9 \
    --ga_mutation_type adaptive2opt \
    --ga_mutation_percent 15 \
    --ga_keep_parents 0 \
    --ga_stop_criteria saturate_30 \
    --ga_seed 42 \
    --ga_k_tournament 3 \
    --ga_two_opt_prob 0.2 \
    --ga_two_opt_max_iter 1
```

Parámetros clave:
- `--threads`: Define los hilos para ACO y GA.

## Estructura del Proyecto
```
/
├── src/
│   ├── aco.py
│   ├── ga.py
│   ├── __init__.py
│   ├── main.py
├── TSPLIB/
├── requirements.txt
├── README.md
```