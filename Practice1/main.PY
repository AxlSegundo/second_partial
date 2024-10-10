from inputs import population
from EDV1 import evo_dif

n = 5
a = 0
b = 10
t_gen = 1500
funct = "exp((1/2) * x) + x**2 - 3*x + 5"
var = "x"
F = 0.8  # Scale factor
CR = 0.7  # Crossover rate

initial_population = population(n, a, b)
final_results = evo_dif(initial_population, t_gen, funct, 'max', F, CR, a, b, var)
print("Resultados finales:", final_results)
