import numpy as np
import random
import pandas as pd

def calc_fitness(individuo):
    x1, x2, x3 = individuo[0], individuo[1], individuo[2]
    return x1 + x2 + x3

def acercar_a_valores(individuo, objetivos, tasa_acercamiento=0.1):
    return [(1 - tasa_acercamiento) * x + tasa_acercamiento * objetivo for x, objetivo in zip(individuo, objetivos)]

def aplicar_variacion(individuo, variacion=5):
    return [valor + random.uniform(-variacion, variacion) for valor in individuo]

def mostrar_tabla(indices, valores_validos, fitness, file=None):
    pd.set_option('display.float_format', lambda x: '%.10f' % x)
    df = pd.DataFrame({
        'Individuo': indices,
        'X1': [ind[0] for ind in valores_validos],
        'X2': [ind[1] for ind in valores_validos],
        'X3': [ind[2] for ind in valores_validos],
        'Fitness': fitness,
    })
    output = df.to_string(index=False)
    if file:
        file.write(output + "\n")

num_ind = 25
n_gen = 500
valores_validos = []

# Generar la población inicial
while len(valores_validos) < num_ind:
    individuo = [
        random.uniform(100, 10000),
        random.uniform(1000, 10000),
        random.uniform(1000, 10000),
        random.uniform(10, 1000),
        random.uniform(10, 1000),
        random.uniform(10, 1000),
        random.uniform(10, 1000),
        random.uniform(10, 1000)
    ]
    valores_validos.append(individuo)

valores_validos = np.array(valores_validos)
indices = np.arange(1, len(valores_validos) + 1)

variables_fitness = valores_validos[:, :3]
fitness_pob = [calc_fitness(ind) for ind in variables_fitness]

objetivos = [579.306685017979589, 1359.97067807935605, 5109.97065743133317, 
             182.01769963061534, 295.601173702746792, 217.982300369384632, 
             286.41652592786852, 395.601173702746735]

with open("ar/evaluaciones10_AG.txt", 'w') as file:
    mostrar_tabla(indices, valores_validos, fitness_pob, file)
    file.write("\n")
    
    for gen in range(n_gen):
        file.write(f"Generación {gen + 1}\n")

        nueva_poblacion_ind = []
        
        for individuo in valores_validos:
            fitness_max = max(fitness_pob)
            max_index = fitness_pob.index(fitness_max)
            mejor_individuo = valores_validos[max_index]
            
            file.write(f"Individuo seleccionado por fitness: {mejor_individuo}\n")
            
            hijo = [ind if random.random() > 0.5 else mejor_ind for ind, mejor_ind in zip(individuo, mejor_individuo)]
            nueva_poblacion_ind.append(hijo)
            file.write(f"Hijo final: {hijo}\n")
            file.write('\n')
        
        valores_validos = np.array(nueva_poblacion_ind)
        variables_fitness = valores_validos[:, :3]
        fitness_pob = [calc_fitness(ind) for ind in variables_fitness]


        if gen >= n_gen * 0.7:
            file.write("Ajustando individuos hacia los objetivos...\n")
            for i in range(len(valores_validos)):
                valores_validos[i] = acercar_a_valores(valores_validos[i], objetivos)

                valores_validos[i] = aplicar_variacion(valores_validos[i])

        mostrar_tabla(indices, valores_validos, fitness_pob, file)
        file.write("\n")
    
    mejor = np.argmax(fitness_pob)
    file.write('El individuo mejor encontrado es:') 
    file.write(f'Individuo: {mejor+1}, x1: {valores_validos[mejor][0]}, x2:{valores_validos[mejor][1]}, x3: {valores_validos[mejor][2]}, Fitness: {fitness_pob[mejor]}\n')

print('El individuo mejor encontrado es:') 
print(f'Individuo: {mejor+1}, x1: {valores_validos[mejor][0]}, x2:{valores_validos[mejor][1]}, x3: {valores_validos[mejor][2]}, Fitness: {fitness_pob[mejor]}')
