from inputs import process_functions
import random
import numpy as np
def comparison(target, trial, opt, a, b):
    if opt == 'max':
        if target < trial and trial <= b:
            return trial
        else:
            return target
    if opt == 'min':
        if target > trial and trial >= a:
            return trial
        else:
            return target

def evo_dif(poblacion, t_gen, funcion, opt, F, CR, a, b, var):
    n = len(poblacion)  # Population size
    best_result = float('inf') if opt == 'min' else float('-inf')

    with open("Practica1/evo_dif_result.txt", "w") as f:
        for gen in range(t_gen):
            f.write(f"Generacion {gen + 1}:\n")
            for i in range(n):
                # Select 3 different indexes
                indices = list(range(n))
                indices.remove(i)  # Do not select the current individual
                a_idx, b_idx, c_idx = set(random.sample(indices, 3))

                # Mutation
                trial = poblacion[a_idx] + F * (poblacion[b_idx] - poblacion[c_idx])
                trial = np.clip(trial, a, b)  # Make sure it is within limits
                R = random.random()

                # Crossover
                if R < CR:
                    new_individual = trial
                else:
                    new_individual = poblacion[i]

                # Evaluate and compare
                trial_result = process_functions(funcion, var, [new_individual])[0]
                current_result = process_functions(funcion, var, [poblacion[i]])[0]

                # Register the procedure (optional)
                #   f.write(f"  - Objetivo: {i+1}: {poblacion[i]}\n")
                #   f.write(f"  - Hijo: {new_individual}\n")
                #   f.write(f"  - Valor de R: {R}\n")

               # Update if the new individual is better
                if comparison(current_result, trial_result, opt, a, b) == trial_result:
                    poblacion[i] = new_individual


                # Update best result
                if (opt == 'min' and trial_result < best_result) or (opt == 'max' and trial_result > best_result):
                    best_result = trial_result

            # Save the population for each generation
            f.write(f"  - Poblacion: {poblacion}\n\n")

    return poblacion
    