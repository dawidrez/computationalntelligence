import time

import pygad


S = [(100,7), (300,7), (200,6), (40,2), (500,5), (70,6), (100,1), (250,3), (300,10), (280,3), (300,15)]

gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(model, solution, solution_idx):
    value_sum = 0
    sum_weight = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            value_sum += S[i][0]
            sum_weight += S[i][1]
    if sum_weight >25:
        return -100
    return value_sum

fitness_function = fitness_func


sol_per_pop = 10
num_genes = len(S)

num_parents_mating = 5
num_generations = 30
keep_parents = 2

parent_selection_type = "sss"

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 5

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
        stop_criteria=["reach_1630"],
        parent_selection_type=parent_selection_type,
        keep_parents=keep_parents,
        crossover_type=crossover_type,
        mutation_type=mutation_type,
        mutation_percent_genes=mutation_percent_genes
    )

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if solution_fitness == BEST_SOLUTION or ga_instance.generations_completed < num_generations:
        n_best_solution += 1
        end = time.time()
        total_time += (end - start)
print(f"Algorithm found {n_best_solution} times best solution")
if n_best_solution:
    print(f"Mean time {total_time/n_best_solution}")


#wyswietlenie wykresu: jak zmieniala sie ocena na przestrzeni pokolen
#ga_instance.plot_fitness()