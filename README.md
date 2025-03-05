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
```sh
python3 -m src.main --tsp TSPLIB/berlin52.tsp --tour TSPLIB/berlin52.opt.tour \
    --threads 32 \
    --aco_ants 10 --aco_iterations 350 \
    --ga_generations 1000 --ga_pop_size 2000 --ga_parents 400 \
    --ga_mutation 0.4 --ga_keep_elite 15 \
    --ga_selection "tournament" --ga_k_tournament 6 --ga_crossover "order" \
    --ga_crossover_prob 0.85 --ga_mutation_type "adaptive" \
    --ga_mutation_percent 12 --ga_keep_parents 0 \
    --ga_stop_criteria "saturate_150" \
    --ga_seed 42
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