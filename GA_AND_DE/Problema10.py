import numpy as np
import random
import pandas as pd

def evaluar_restricciones(x1, x2, x3, x4, x5, x6, x7, x8):
    g1 = -1 + 0.0025 * (x4 + x6)
    g2 = -1 + 0.0025 * (x5 + x7 - x4)
    g3 = -1 + 0.01 * (x8 - x5)
    g4 = -x1 * x6 + 833.33252 * x4 + 100 * x1 - 83333.333
    g5 = -x2 * x7 + 1250 * x5 + x2 * x4 - 1250 * x4
    g6 = -x3 * x8 + 1250000 + x3 * x5 - 2500 * x5
    return [g1, g2, g3, g4, g5, g6]

def verificar_restricciones(individuo, tolerancia=1e-6):
    restricciones = evaluar_restricciones(*individuo)
    return all(r <= tolerancia for r in restricciones)

def calc_fitness(individuo):
    x1, x2, x3 = individuo[0], individuo[1], individuo[2]
    return x1 + x2 + x3

def mostrar_tabla(indices, valores_validos, restricciones_vals, fitness, file=None):
    pd.set_option('display.float_format', lambda x: '%.10f' % x)
    df = pd.DataFrame({
        'Individuo': indices,
        'X1': [ind[0] for ind in valores_validos],
        'X2': [ind[1] for ind in valores_validos],
        'X3': [ind[2] for ind in valores_validos],
        'X4': [ind[3] for ind in valores_validos],
        'X5': [ind[4] for ind in valores_validos],
        'X6': [ind[5] for ind in valores_validos],
        'X7': [ind[6] for ind in valores_validos],
        'X8': [ind[7] for ind in valores_validos],
        'Fitness': fitness,
        'Eval_Rest': restricciones_vals
    })

    # Mostrar la tabla
    output = df.to_string(index=False)
    if file:
        file.write(output + "\n")  # Guardar en archivo

# Crear población aleatoria y verificar restricciones
num_ind = 25  # Número de individuos en la población
F = 0.9
CR = 0.9
n_gen = 500
valores_validos = []

# Generar individuos aleatoriamente y verificar restricciones
while len(valores_validos) < num_ind:
    individuo = [
        random.uniform(100, 10000),  # x1
        random.uniform(1000, 10000),  # x2
        random.uniform(1000, 10000),  # x3
        random.uniform(10, 1000),     # x4
        random.uniform(10, 1000),     # x5
        random.uniform(10, 1000),     # x6
        random.uniform(10, 1000),     # x7
        random.uniform(10, 1000)      # x8
    ]
    
    # Verificar si el individuo cumple con todas las restricciones
    if verificar_restricciones(individuo):
        valores_validos.append(individuo)

# Convertir la lista de valores válidos en un array para manipulación
valores_validos = np.array(valores_validos)

# Evaluamos las restricciones para los valores válidos
restricciones_vals = [evaluar_restricciones(*ind) for ind in valores_validos]

# Creamos los índices de los individuos
indices = np.arange(1, len(valores_validos) + 1)

# Calcular el fitness
fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]

# Guardar información en un archivo txt
with open("bioinspirados/trabajos/evaluaciones10_ED.txt", 'w') as file:
    # Guardar tabla    
    mostrar_tabla(indices, valores_validos, restricciones_vals, fitness_pob, file)
    file.write("\n")
    
    # Ciclo de generaciones
    for gen in range(n_gen):
        file.write(f"Generación {gen + 1}\n")
        print(f"Generación {gen + 1}")
        
        nueva_poblacion_ind = []  # Arreglo para almacenar la nueva población de esta generación
        
        for indice, individuo in enumerate(valores_validos):
            indices_disponibles = list(range(valores_validos.shape[0]))
            indices_disponibles.remove(indice)  # Excluimos el índice que no queremos
            indices_seleccionados = list(random.sample(indices_disponibles, 3))  # Selección aleatoria
            xa, xb, xc = indices_seleccionados  # Extraemos los índices como elementos 
        
            file.write(f'Target {indice+1}, x{indice+1}: {individuo}\n')
            # Hijo
            file.write(f"xa: {valores_validos[xa]}, xb: {valores_validos[xb]}, xc:{valores_validos[xc]}\n")
            u = valores_validos[xa] + F * (valores_validos[xb] - valores_validos[xc])  # Fórmula de mutación
            file.write(f"Valor del hijo: {u}\n")
            
            u[0] = np.clip(u[0], 100, 10000)  # x1
            u[1] = np.clip(u[1], 1000, 10000)  # x2
            u[2] = np.clip(u[2], 1000, 10000)  # x3
            u[3] = np.clip(u[3], 10, 1000)     # x4
            u[4] = np.clip(u[4], 10, 1000)     # x5
            u[5] = np.clip(u[5], 10, 1000)     # x6
            u[6] = np.clip(u[6], 10, 1000)     # x7
            u[7] = np.clip(u[7], 10, 1000)     # x8
            
            # Verificar restricciones del hijo mutado
            if not verificar_restricciones(u):
                file.write(f"Se necesita reparar a {u}\n")
                while not verificar_restricciones(u):
                    u = [
                        random.uniform(100, 10000),  # x1
                        random.uniform(1000, 10000),  # x2
                        random.uniform(1000, 10000),  # x3
                        random.uniform(10, 1000),     # x4
                        random.uniform(10, 1000),     # x5
                        random.uniform(10, 1000),     # x6
                        random.uniform(10, 1000),     # x7
                        random.uniform(10, 1000)      # x8
                    ]
            file.write(f"El hijo se ha reparado: {u}\n")

            num_random = random.random()
            file.write(f"Número random generado: {num_random}\n")
            
            if num_random > CR:
                file.write(f"El número random {num_random} es mayor al CR, se toma al padre {individuo} \n")
                u = individuo  # El hijo toma el valor del padre si CR es menor
                nueva_poblacion_ind.append(u)
            else:
                file.write('El número random es menor o igual a CR, se considera al hijo \n')
                fitness_target = fitness_pob[indice]
                fitness_trial = calc_fitness(u)
                file.write(f"El fitness del padre es: {fitness_target} y el fitness del hijo es: {fitness_trial}\n")
                
                if fitness_trial < fitness_target:  # Si el hijo es mejor que el padre
                    seleccion = u
                    file.write(f"Se selecciona al hijo {seleccion}\n")
                else:
                    seleccion = individuo
                    file.write(f"Se selecciona al padre {seleccion}\n")
                nueva_poblacion_ind.append(seleccion)
                
            file.write('\n')
        
        # Actualizar población
        valores_validos = np.array(nueva_poblacion_ind)
        
        # Evaluar restricciones para la nueva generación
        restricciones_vals = [evaluar_restricciones(*ind) for ind in valores_validos]
        
        # Calcular el fitness para la nueva generación
        fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]
        mostrar_tabla(indices, valores_validos, restricciones_vals, fitness_pob, file)
        file.write("\n")
        
    mejor = np.argmin(fitness_pob)    
    file.write('El individuo mejor encontrado es:') 
    file.write(f'Individuo: {mejor+1}, x1: {valores_validos[mejor][0]}, x2:{valores_validos[mejor][1]}, x3: {valores_validos[mejor][2]}, Fitness: {fitness_pob[mejor]}, Restricciones: {restricciones_vals[mejor]}\n')

print('El individuo mejor encontrado es:') 
print(f'Individuo: {mejor+1}, x1: {valores_validos[mejor][0]}, x2:{valores_validos[mejor][1]}, x3: {valores_validos[mejor][2]}, Fitness: {fitness_pob[mejor]}, Restricciones: {restricciones_vals[mejor]}')
