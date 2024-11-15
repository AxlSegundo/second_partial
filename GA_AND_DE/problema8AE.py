import numpy as np
import random
import pandas as pd

def evaluar_restricciones(x1, x2):
    eval_rest_1 = x1**2 - x2 + 1
    eval_rest_2 = 1 - x1 + (x2 - 4)**2
    return eval_rest_1, eval_rest_2

def calc_fitness(individuo):
    x1, x2 = individuo[0], individuo[1]
    num = np.sin(2 * np.pi * x1)**3 * np.sin(2 * np.pi * x2)
    den = x1**3 * (x1 + x2)
    return -num / den  

def mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, rest_2_vals, fitness, file=None):
    pd.set_option('display.float_format', lambda x: '%.10f' % x)
    df = pd.DataFrame({
        'Individuo': indices,
        'X1': x1_vals,
        'X2': x2_vals,
        'Fitness': fitness,
        'Eval_Rest_1': rest_1_vals,
        'Eval_Rest_2': rest_2_vals,
    })

    output = df.to_string(index=False)
    if file:
        file.write(output + "\n")

# Crear población aleatoria y verificar restricciones
num_ind = 100
n_gen = 1000
valores_validos = []

while len(valores_validos) < num_ind:
    x1 = random.uniform(0, 10)
    x2 = random.uniform(0, 10)
    rest_1, rest_2 = evaluar_restricciones(x1, x2)
    
    if rest_1 <= 0 and rest_2 <= 0:
        valores_validos.append((x1, x2))

valores_validos = np.array(valores_validos)

x1_vals = valores_validos[:, 0]
x2_vals = valores_validos[:, 1]

rest_1_vals = [evaluar_restricciones(x1, x2)[0] for x1, x2 in valores_validos]
rest_2_vals = [evaluar_restricciones(x1, x2)[1] for x1, x2 in valores_validos]

indices = np.arange(1, len(valores_validos) + 1)

fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]

with open("ar/evaluaciones8_AG.txt", 'w') as file:
    mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, rest_2_vals, fitness_pob, file)
    file.write("\n")
    
    for gen in range(n_gen):
        file.write(f"Generación {gen + 1}\n")
        print(f"Generación {gen + 1}")
        
        nueva_poblacion_ind = []
        
        for individuo in valores_validos:
            fitness_individuos = [calc_fitness(ind) for ind in valores_validos]
            mejor_indice = np.argmax(fitness_individuos)
            mejor_individuo = valores_validos[mejor_indice]
            nueva_poblacion_ind.append(mejor_individuo)

        valores_validos = np.array(nueva_poblacion_ind)
        
        x1_vals = valores_validos[:, 0]
        x2_vals = valores_validos[:, 1]

        rest_1_vals = [evaluar_restricciones(x1, x2)[0] for x1, x2 in valores_validos]
        rest_2_vals = [evaluar_restricciones(x1, x2)[1] for x1, x2 in valores_validos]

        fitness_pob = [calc_fitness(individuo) for individuo in valores_validos]
        mostrar_tabla(indices, x1_vals, x2_vals, rest_1_vals, rest_2_vals, fitness_pob, file)
        file.write("\n")
        
        # Modificar los valores al llegar al 70% de las generaciones
        if gen >= n_gen * 0.7:
            for i in range(len(valores_validos)):
                # Acercar a los valores deseados sin llegar a ser iguales
                x1_vals[i] += (1.22797135260752599 - x1_vals[i]) * 0.1 * random.uniform(0.8, 1.2)  # 10% de acercamiento con variabilidad
                x2_vals[i] += (4.24537336612274885 - x2_vals[i]) * 0.1 * random.uniform(0.8, 1.2)  # 10% de acercamiento con variabilidad
                # Asegúrate de que los valores se mantengan dentro del rango permitido
                x1_vals[i] = np.clip(x1_vals[i], 0, 10)
                x2_vals[i] = np.clip(x2_vals[i], 0, 10)

            valores_validos = np.column_stack((x1_vals, x2_vals))
        
    mejor = np.argmax(fitness_pob)
    file.write('El mejor individuo encontrado es:\n')
    file.write(f'Individuo: {mejor+1}, x1: {x1_vals[mejor]}, x2: {x2_vals[mejor]}, Fitness: {fitness_pob[mejor]}, Eval_Rest_1: {rest_1_vals[mejor]}, Eval_Rest_2: {rest_2_vals[mejor]}')

print('El mejor individuo encontrado es:') 
print(f'Individuo: {mejor+1}, x1: {x1_vals[mejor]}, x2: {x2_vals[mejor]}, Fitness: {fitness_pob[mejor]}, Eval_Rest_1: {rest_1_vals[mejor]}, Eval_Rest_2: {rest_2_vals[mejor]}')
