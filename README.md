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
python -m src.main --threads [NUM_THREADS]
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