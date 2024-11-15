import sympy as sp
import random

# Function to process symbolic expressions with multiple variables
def process_functions(expresion_str, variables_str, valores):
    variables = sp.symbols(variables_str)
    expresion = sp.sympify(expresion_str)
    funcion = sp.lambdify(variables, expresion, modules=["numpy"])
    resultados = [funcion(*valor) for valor in valores]
    return resultados

# Function to check if a set of values satisfies all constraints
def check_constraints(restricciones_str, variables_str, valores):
    variables = sp.symbols(variables_str)
    restricciones = [sp.sympify(r) for r in restricciones_str]

    for valor in valores:
        # Create a dictionary for variable substitution
        sustituciones = dict(zip(variables, valor))
        # Evaluate each restriction as a logical expression
        if not all(restriccion.subs(sustituciones) for restriccion in restricciones):
            return False  # If any restriction is not satisfied, return False
    return True  # If all restrictions are satisfied, return True

# Function to generate a population of random values with n dimensions
def population(n, d, a, b):
    popl = []
    for i in range(n):
        individuo = [round(random.uniform(a, b), 8) for _ in range(d)]
        popl.append(individuo)
    return popl
