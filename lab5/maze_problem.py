import numpy as np
import math
import pygad
import time
maze = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])

gene_space = [1,2,3,4] # 1 left 2 up 3 right 4 down


def fitness_func(model, solution, solution_idx):
    x, y = 1,1
    for i in range(0, len(solution)):
        if solution[i] == 1:
            x -= 1
        elif solution[i] == 2:
            y -= 1
        elif solution[i] == 3:
            x += 1
        else:
            y += 1
        if maze[x][y] == 1:
            return -math.sqrt(((10-x)**2+(10-y)**2))
        if x == 10 and y == 10:
            return 1000
    return -math.sqrt(((10-x)**2+(10-y)**2))




fitness_function = fitness_func


sol_per_pop = 1000
num_genes = 30

num_parents_mating = 50
num_generations = 100
keep_parents = 5

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 10

n_iterations = 10
BEST_SOLUTION =1630
n_best_solution = 0
total_time = 0
for i in range(n_iterations):
    start = time.time()
    ga_instance = pygad.GA(
            gene_space=gene_space,
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            fitness_func=fitness_function,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            stop_criteria=["reach_1000"],
            parent_selection_type=parent_selection_type,
            keep_parents=keep_parents,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes
        )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if solution_fitness == 1000 or ga_instance.generations_completed < num_generations:
        n_best_solution +=1
        end = time.time()
        total_time += (end - start)
print(f"Algorithm found {n_best_solution} times best solution")
if n_best_solution:
    print(f"Mean time {total_time / n_best_solution}")
