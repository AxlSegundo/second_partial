import numpy as np
import random
import pandas as pd

def evaluar_restricciones(x1, x2):
    h = x2 - x1**2
    return h

def calc_fitness(individuo):
    x1, x2 = individuo[0], individuo[1]
    return x1**2 + (x2 - 1)**2


def mostrar_tabla(indices, x1_vals, x2_vals,rest_1_vals,fitness,file=None):
    pd.set_option('display.float_format', lambda x: '%.10f' % x)
    df = pd.DataFrame({
        'Individuo': indices,
        'X1': x1_vals,
        'X2': x2_vals,
        'Fitnes': fitness,
        'Eval_Rest_1': rest_1_vals
    })

    # Mostrar la tabla
    output = df.to_string(index=False)
    if file:
        file.write(output + "\n")  # Guardar en archivo

# Crear población aleatoria y verificar restricciones
num_ind = 10  # Número de individuos en la población
F = 0.5
CR = 0.5
n_gen = 500
valores_validos = []
tolerancia = 1e-6

# Generar individuos aleatoriamente y verificar restricciones
while len(valores_validos) < num_ind:
    # Generar valores aleatorios para x1 y x2 dentro de los rangos
    x1 = random.uniform(-1, 1)  # Rango de x1
    x2 = random.uniform(-1, 1)      # Rango de x2
    
    # Si el individuo cumple con ambas restricciones, se agrega a la población
    rest_1 = evaluar_restricciones(x1, x2) 
    # Si el individuo cumple con la restricción de igualdad dentro de la tolerancia
    if abs(rest_1) <= tolerancia:
        valores_validos.append((x1, x2))

# Convertir la lista de valores válidos en un array para manipulación
valores_validos = np.array(valores_validos)

# Extraemos los valores de x1 y x2 de los puntos válidos
x1_vals = valores_validos[:, 0]
x2_vals = valores_validos[:, 1]

# Evaluamos las restricciones para los valores válidos
rest_1_vals = []
for i in range(len(x1_vals)):
    rest_1 = evaluar_restricciones(x1_vals[i], x2_vals[i])
    rest_1_vals.append(rest_1)

# Creamos los índices de los individuos
indices = np.arange(1, len(valores_validos) + 1)

# Calcular el fitness
fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]

# Guardar información en un archivo txt
with open("bioinspirados/trabajos/evaluaciones11_ED.txt", 'w') as file:
    # Guardar tabla    
    mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, fitness_pob, file)
    file.write("\n")
    
    # Ciclo de generaciones
    for gen in range(n_gen):
        file.write(f"Generación {gen + 1}\n")
        print(f"Generación {gen + 1}")
        
        nueva_poblacion_ind = []  # Arreglo para almacenar la nueva población de esta generación
        
        for indice, individuo in enumerate(valores_validos):
            indices_disponibles = list(range(valores_validos.shape[0]))
            indices_disponibles.remove(indice)      # Excluimos el índice que no queremos
            indices_seleccionados = list(random.sample(indices_disponibles, 3))   # Creamos una lista con los índices
            xa, xb, xc = indices_seleccionados  # Extraemos los índices como elementos 
        
            file.write(f'Target {indice+1}, x{indice+1}: {individuo}\n')
            # Hijo
            file.write(f"xa: {valores_validos[xa]}, xb: {valores_validos[xb]}, xc:{valores_validos[xc]}\n")
            u = valores_validos[xa] + F * (valores_validos[xb] - valores_validos[xc])  # Fórmula de mutación
            file.write(f"Valor del hijo: {u}\n")
            eval_r1 = evaluar_restricciones(u[0], u[1])
            
            # Revisa si necesita ser reparado y reparar
            if abs(eval_r1) > tolerancia:
                file.write(f"Se necesita reparar a {u}\n")
                # Reparar ajustando u[1] para que cumpla con la restricción
                u[1] = u[0]**2
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
        
        # Mostrar tabla de la nueva generación
        x1_vals = valores_validos[:, 0]
        x2_vals = valores_validos[:, 1]

        rest_1_vals = []
        for i in range(len(x1_vals)):
            rest_2 = evaluar_restricciones(x1_vals[i], x2_vals[i])
            rest_1_vals.append(rest_1)

        fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]
        mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, fitness_pob, file)
        file.write("\n")
        
    mejor = np.argmin(fitness_pob)    
    file.write('El individuo mejor escontrado es:') 
    file.write(f'Individuo: {mejor+1}, x1: {x1_vals[mejor]}, x2:{x2_vals[mejor]},Fitness: {fitness_pob[mejor]},Valor de restrición 1: {rest_1_vals[mejor]}')

print('El individuo mejor escontrado es:') 
print(f'Individuo: {mejor+1}, x1: {x1_vals[mejor]}, x2:{x2_vals[mejor]},Fitness: {fitness_pob[mejor]},Valor de restrición 1: {rest_1_vals[mejor]}')
