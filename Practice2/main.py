from inputs import population
from EDV2 import evo_dif

n = 4  # Number of individuals
d = 2  # Number of dimensions (variables)
a = 0
b= 100

t_gen = 500
funct = "(x - 10)**3 + (y -20)**3"
var = "x y"
restricciones = ["-1*(x - 5)**2 -1*(y - 5)**2 + 100 <=0", "(x - 6)**2 + (x2 -5)**2 - 82.81 <=0"]
F = 0.7  # Scale factor
CR = 0.5  # Crossover rate

# Generate initial population with n individuals and d dimensions
initial_population = population(n, d, a, b)
# Execute the differential evolution algorithm
final_results = evo_dif(initial_population, t_gen, funct, 'max', F, CR, a, b, var, restricciones)

print("Resultados finales:", final_results)

