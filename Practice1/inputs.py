import sympy as sp
import random

# Function to process symbolic expressions
def process_functions(expresion_str, variables_str, valores):
    # Define the variables
    variables = sp.symbols(variables_str)
    
    # Convert the expression to symbolic
    expresion = sp.sympify(expresion_str)
    
    # Create an evaluable function with lambdify
    funcion = sp.lambdify(variables, expresion, modules=["numpy"])
    
    # Evaluate the function for each value in the population list
    resultados = [funcion(valor) for valor in valores]
    
    return resultados

# Function to generate a population of random values
def population(n, a, b):
    popl = []
    for i in range(n):
        p = round(random.uniform(a, b), 8)  # Round to 8 decimal places
        popl.append(p)
    return popl

