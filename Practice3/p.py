import numpy as np

# Función objetivo
def funcion_objetivo(x):
    return (x[0] - 10)**3 + (x[1] - 20)**3

# Restricciones
def restricciones(x):
    g1 = -(x[0] - 5)**2 - (x[1] - 5)**2 + 100
    g2 = (x[0] - 6)**2 + (x[1] - 5)**2 - 82.81
    return g1 <= 0 and g2 <= 0

# Método de proyección para reparar individuos fuera del rango permitido
def proyectar_individuo(x, limites):
    for i in range(len(x)):
        x[i] = np.clip(x[i], limites[i][0], limites[i][1])
    return x

# Algoritmo de Evolución Diferencial mejorado con ajuste cercano al resultado óptimo
def evolucion_diferencial(funcion_objetivo, limites, num_individuos=50, max_generaciones=1500, F=0.8, CR=0.5, generaciones_estables=50, archivo_log="evolucion_diferencial.txt"):
    # Generación inicial de la población
    poblacion = np.random.uniform([limite[0] for limite in limites], 
                                  [limite[1] for limite in limites], 
                                  (num_individuos, len(limites)))
    
    mejor_valor_global = float('inf')
    generaciones_sin_mejora = 0
    resultado_optimo = (14.095, 0.8429607892154796)
    perturbacion_factor = 0.08 # Factor de perturbación para explorar valores cercanos
    
    # Abrir el archivo para registrar el progreso
    with open('Practice3/archivo_log.txt', "w") as log:
        log.write("Generación,Mejor Individuo,Valor de la Función Objetivo\n")

        for generacion in range(max_generaciones):
            nueva_poblacion = np.copy(poblacion)

            for i in range(num_individuos):
                # Selección aleatoria de tres individuos diferentes
                indices = list(range(num_individuos))
                indices.remove(i)
                a, b, c = poblacion[np.random.choice(indices, 3, replace=False)]

                # Mutación
                mutante = a + F * (b - c)
                mutante = proyectar_individuo(mutante, limites)

                # Cruce
                trial = np.copy(poblacion[i])
                for j in range(len(limites)):
                    if np.random.rand() < CR:
                        trial[j] = mutante[j]

                # Proyección del individuo prueba si no cumple restricciones
                trial = proyectar_individuo(trial, limites)

                # Evaluar función objetivo y aplicar restricciones
                if restricciones(trial):
                    if funcion_objetivo(trial) < funcion_objetivo(poblacion[i]):
                        nueva_poblacion[i] = trial

            # Actualizar la población
            poblacion = nueva_poblacion

            # Encontrar el mejor individuo de la generación actual
            mejor_individuo = min(poblacion, key=funcion_objetivo)
            mejor_valor = funcion_objetivo(mejor_individuo)

            # Escribir la generación actual en el archivo
            log.write(f"{generacion},{mejor_individuo},{mejor_valor}\n")

            # Verificar si el nuevo mejor valor mejora al mejor global
            if mejor_valor < mejor_valor_global:
                mejor_valor_global = mejor_valor
                generaciones_sin_mejora = 0
            else:
                generaciones_sin_mejora += 1

            # Si se alcanza el 70% de las generaciones, perturbar el mejor individuo
            if generacion >= 0.7 * max_generaciones:
                perturbacion = np.random.uniform(-perturbacion_factor, perturbacion_factor, size=len(limites))
                mejor_individuo_perturbado = mejor_individuo + perturbacion
                mejor_individuo_perturbado = proyectar_individuo(mejor_individuo_perturbado, limites)
                
                # Reemplazar el peor individuo con la versión perturbada si es mejor
                peor_individuo_index = np.argmax([funcion_objetivo(ind) for ind in poblacion])
                if funcion_objetivo(mejor_individuo_perturbado) < funcion_objetivo(poblacion[peor_individuo_index]):
                    poblacion[peor_individuo_index] = mejor_individuo_perturbado

            # Condición de cierre si se alcanza el resultado óptimo o se aproxima por 50 generaciones
            if np.isclose(mejor_valor, resultado_optimo[1], atol=1e-5) and np.allclose(mejor_individuo, resultado_optimo[0], atol=1e-5):
                generaciones_sin_mejora += 1
                if generaciones_sin_mejora >= generaciones_estables:
                    log.write(f"Condición de cierre alcanzada en la generación {generacion}.\n")
                    print(f"Condición de cierre alcanzada en la generación {generacion}.")
                    break

    return mejor_individuo, mejor_valor_global

# Parámetros del problema
limites = [(13, 100), (0, 100)]

# Ejecutar el algoritmo
mejor_individuo, mejor_valor = evolucion_diferencial(funcion_objetivo, limites)
print(f"Mejor solución encontrada: x = {mejor_individuo}")
print(f"Valor de la función objetivo en la mejor solución: f(x) = {mejor_valor}")
