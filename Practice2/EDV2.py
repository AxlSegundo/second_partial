from inputs import process_functions, check_constraints
import random
import numpy as np

def comparison(target, trial, opt, a, b):
    if opt == 'max':
        return trial if np.all(trial <= b) and trial > target else target
    if opt == 'min':
        return trial if np.all(trial >= a) and trial < target else target

def evo_dif(poblacion, t_gen, funcion, opt, F, CR, a, b, var, restricciones):
    n = len(poblacion)  # Population size
    d = len(poblacion[0])  # Number of dimensions

    with open("Practice2/evo_dif_result.txt", "w") as f:
        for gen in range(t_gen):
            f.write(f"Generación {gen + 1}:\n")
            for i in range(n):
                # Select 3 different indexes
                indices = list(range(n))
                indices.remove(i)  # Do not select the current individual
                a_idx, b_idx, c_idx = random.sample(indices, 3)

                # Mutation
                trial = [poblacion[a_idx][dim] + F * (poblacion[b_idx][dim] - poblacion[c_idx][dim]) for dim in range(d)]
                
                # Crossover
                R = random.random()
                new_individual = [trial[dim] if R < CR else poblacion[i][dim] for dim in range(d)]

                # Check constraints
                if not check_constraints(restricciones, var, [new_individual]):
                    new_individual = poblacion[i]

                # Evaluate and compare
                trial_result = process_functions(funcion, var, [new_individual])[0]
                current_result = process_functions(funcion, var, [poblacion[i]])[0]

                # Register the procedure (optional)
                f.write(f"  - Objetivo {i+1}: {poblacion[i]}\n")
                f.write(f"  - Valor de objetivo: {current_result}\n")
                f.write(f"  - Hijo: {new_individual}\n")
                f.write(f"  - Valor de hijo: {trial_result}\n")
                f.write(f"  - Valor de R: {R}\n")

                # Update if the new individual is better
                if comparison(current_result, trial_result, opt, a, b) == trial_result:
                    poblacion[i] = new_individual

            # Save the population for each generation
            f.write(f"  - Población: {poblacion}\n\n")

    return poblacion
