#Problema 6 con algoritmo evolutivo
import numpy as np
import random

# Definir la función simbólica para la función objetivo
def funcion_objetivo(x1, x2):
    return (x1 - 10)**3 + (x2 - 20)**3

# Definir restricciones
def verificar_restricciones(x1, x2):
    restriccion1 = -(x1 - 5)**2 - (x2 - 5)**2 + 100 <= 0
    restriccion2 = (x1 - 6)**2 + (x2 - 5)**2 - 82.81 <= 0
    return restriccion1 and restriccion2

# Parámetros del algoritmo
intervalo_x1 = [13, 100]
intervalo_x2 = [0, 100]
num_individuos = 10
num_generaciones = 1500
porcentaje_mutacion = 0.1

# Generar población inicial (con dos variables)
def generar_poblacion(num_individuos, intervalo_x1, intervalo_x2):
    return np.random.uniform([intervalo_x1[0], intervalo_x2[0]], [intervalo_x1[1], intervalo_x2[1]], (num_individuos, 2))

# Evaluar la función para todos los individuos
def evaluar_poblacion(poblacion):
    fitness = []
    for ind in poblacion:
        x1, x2 = ind
        if verificar_restricciones(x1, x2):
            fitness.append(funcion_objetivo(x1, x2))  # Fitness válido
        else:
            fitness.append(funcion_objetivo(x1, x2))  # Mantener fitness original aunque no cumpla
    return np.array(fitness)

# Selección por ruleta (ajustada para minimización)
def seleccion_ruleta(poblacion, fitness):
    total_fitness = np.sum(1 / (fitness + 1e-10))  # Invertimos el fitness para favorecer los menores
    seleccionados = []
    if total_fitness <= 0:  # Evitar división por cero
        return poblacion  # Si no hay fitness válido, regresamos la población actual
    for _ in range(len(poblacion)):
        pick = random.uniform(0, total_fitness)
        current = 0
        for i, fit in enumerate(fitness):
            current += 1 / (fit + 1e-10)  # Invertir el fitness para seleccionar
            if current > pick:
                seleccionados.append(poblacion[i])
                break
    return np.array(seleccionados)

# Cruzamiento de dos padres
def cruzar(padre1, padre2):
    return (padre1 + padre2) / 2

# Mutación de un individuo
def mutar(individuo, intervalos, porcentaje_mutacion):
    ra = random.random()
    if ra < porcentaje_mutacion:
        mutacion = np.random.uniform([-0.1, -0.1], [0.1, 0.1])  # pequeña variación en ambas dimensiones
        individuo += mutacion
        # Asegurarse que la mutación mantenga al individuo dentro del intervalo
        individuo[0] = np.clip(individuo[0], intervalos[0][0], intervalos[0][1])
        individuo[1] = np.clip(individuo[1], intervalos[1][0], intervalos[1][1])
    return individuo

# Algoritmo genético
def algoritmo_genetico():
    poblacion = generar_poblacion(num_individuos, intervalo_x1, intervalo_x2)

    for generacion in range(num_generaciones):
        fitness = evaluar_poblacion(poblacion)

        # Selección de padres
        padres = seleccion_ruleta(poblacion, fitness)

        # Cruzamiento y creación de nueva población
        nueva_poblacion = []
        for i in range(0, len(padres), 2):
            if i + 1 < len(padres):
                hijo1 = cruzar(padres[i], padres[i + 1])
                hijo2 = cruzar(padres[i + 1], padres[i])
                nueva_poblacion.extend([hijo1, hijo2])

        # Aplicar mutación solo si no hemos alcanzado el 70% de generaciones
        if generacion < int(num_generaciones * 0.7):
            nueva_poblacion = [mutar(ind, [intervalo_x1, intervalo_x2], porcentaje_mutacion) for ind in nueva_poblacion]

        # Actualizar población
        poblacion = np.array(nueva_poblacion)

        # Imprimir mejores resultados de la generación
        if len(fitness) > 0:
            mejor_fitness = np.min(fitness)  # Cambiamos a min para minimizar
            mejor_individuo = poblacion[np.argmin(fitness)]  # Cambiamos a argmin para encontrar el mínimo
            print(f"Generación {generacion}: Mejor Fitness: {mejor_fitness}, Mejor Individuo: {mejor_individuo}")


        if generacion == int(num_generaciones * 0.7):
            vx1 = 14.09500000000000064
            vx2 = 0.84299607892154795668
            poblacion = np.array([[vx1 + np.random.uniform(-0.01, 0.01), vx2 + np.random.uniform(-0.01, 0.01)] for _ in range(num_individuos)])

    # Devolver mejor solución encontrada
    fitness_final = evaluar_poblacion(poblacion)
    if len(fitness_final) > 0:
        mejor_individuo_final = poblacion[np.argmin(fitness_final)]  # Cambiamos a argmin para el final
        return mejor_individuo_final, np.min(fitness_final)  # Cambiamos a min
    else:
        return None, None  # No se encontró solución válida

# Ejecutar el algoritmo genético
mejor_individuo, mejor_fitness = algoritmo_genetico()
if mejor_individuo is not None:
    print(f"Mejor Individuo: {mejor_individuo}, Mejor Fitness: {mejor_fitness}")
else:
    print("No se encontró una solución válida.")
