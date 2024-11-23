import math
import time

import pygad

gene_space = {'low': 0, 'high': 1}

def endurance(x,y,z,u,v,w):
    return math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u) +math.cos(v*w)

def fitness_func(model, solution, solution_idx):
    return endurance(solution[0], solution[1], solution[2], solution[3], solution[4], solution[5])

fitness_function = fitness_func


sol_per_pop = 10
num_genes = 6

num_parents_mating = 5
num_generations = 30
keep_parents = 2

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 3

n_iterations = 10
n_best_solution = 0
total_time = 0
start = time.time()
ga_instance = pygad.GA(
        gene_space=gene_space,
        num_generations=num_generations,
        num_parents_mating=num_parents_mating,
        fitness_func=fitness_function,
        sol_per_pop=sol_per_pop,
        num_genes=num_genes,

        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes
    )

ga_instance.run()

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(solution, solution_fitness, solution_idx)


#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
ga_instance.plot_fitness()