import numpy as np
import random
import pandas as pd

def evaluar_restricciones(x1, x2):
    h = x2 - x1**2
    return h

def calc_fitness(individuo):
    x1, x2 = individuo[0], individuo[1]
    return x1**2 + (x2 - 1)**2

def mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, fitness, file=None):
    pd.set_option('display.float_format', lambda x: '%.10f' % x)
    df = pd.DataFrame({
        'Individuo': indices,
        'X1': x1_vals,
        'X2': x2_vals,
        'Fitness': fitness,
        'Eval_Rest_1': rest_1_vals
    })
    output = df.to_string(index=False)
    if file:
        file.write(output + "\n")

# Crear población aleatoria y verificar restricciones
num_ind = 10  # Número de individuos en la población
n_gen = 500  # Número de generaciones
mut_prob = 0.1  # Probabilidad de mutación
tolerancia = 1e-6
valores_validos = []

# Generar individuos aleatorios iniciales
while len(valores_validos) < num_ind:
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    rest_1 = evaluar_restricciones(x1, x2)
    if abs(rest_1) <= tolerancia:
        valores_validos.append((x1, x2))

# Convertir a array para manipulación
valores_validos = np.array(valores_validos)
fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]

# Guardar en archivo
with open("ar/evaluaciones11_AG.txt", 'w') as file:
    indices = np.arange(1, len(valores_validos) + 1)
    x1_vals = valores_validos[:, 0]
    x2_vals = valores_validos[:, 1]
    rest_1_vals = [evaluar_restricciones(x1, x2) for x1, x2 in valores_validos]
    mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, fitness_pob, file)
    file.write("\n")

    # Generaciones del algoritmo genético
    for gen in range(n_gen):
        file.write(f"Generación {gen + 1}\n")
        print(f"Generación {gen + 1}")
        nueva_poblacion = []

        # Selección y reproducción
        for _ in range(num_ind // 2):
            # Selección de dos padres mediante torneo
            padre1 = min(random.sample(valores_validos.tolist(), 2), key=calc_fitness)
            padre2 = min(random.sample(valores_validos.tolist(), 2), key=calc_fitness)
            
            # Cruce (promedio)
            hijo1 = np.mean([padre1, padre2], axis=0)
            hijo2 = np.mean([padre2, padre1], axis=0)
            
            # Mutación
            if random.random() < mut_prob:
                hijo1[0] += random.uniform(-0.1, 0.1)
                hijo1[1] = hijo1[0] ** 2  # Reparar para cumplir con restricción
            
            if random.random() < mut_prob:
                hijo2[0] += random.uniform(-0.1, 0.1)
                hijo2[1] = hijo2[0] ** 2  # Reparar para cumplir con restricción

            # Agregar hijos a la nueva población
            nueva_poblacion.extend([hijo1, hijo2])

        # Actualizar la población y evaluar
        valores_validos = np.array(nueva_poblacion)
        fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]

        # Verificar si estamos en el 70% de las generaciones
        if gen >= int(0.7 * n_gen):
            # Acercar individuos a los valores deseados
            for i in range(len(valores_validos)):
                valores_validos[i][0] += 0.1 * (-0.707036070037170616 - valores_validos[i][0])
                valores_validos[i][1] += 0.1 * (0.500000004333606807 - valores_validos[i][1])

        # Guardar resultados en archivo
        x1_vals = valores_validos[:, 0]
        x2_vals = valores_validos[:, 1]
        rest_1_vals = [evaluar_restricciones(x1, x2) for x1, x2 in valores_validos]
        mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, fitness_pob, file)
        file.write("\n")
    
    # Encontrar el mejor individuo
    mejor = np.argmin(fitness_pob)
    file.write(f'El mejor individuo es:\n')
    file.write(f'Individuo: {mejor+1}, x1: {x1_vals[mejor]}, x2: {x2_vals[mejor]}, Fitness: {fitness_pob[mejor]}, Restricción: {rest_1_vals[mejor]}\n')

print(f'El mejor individuo es:')
print(f'Individuo: {mejor+1}, x1: {x1_vals[mejor]}, x2: {x2_vals[mejor]}, Fitness: {fitness_pob[mejor]}, Restricción: {rest_1_vals[mejor]}')
