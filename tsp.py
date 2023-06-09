import random
import numpy as np

def generate_tsp_instance(num_customers):
    coordinates = np.random.randint(0, 100, size=(num_customers, 2))
    return coordinates



def euclidean_distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


#TODO: implement your tsp_fitness() fucntion
def tsp_fitness(solution, coordinates):
    '''
    @description: evaluate an individual
    @param: solution:  list of nodes, e.g., [3, 2, 4, 1]
    @param: coordinates: coordinates of each customer, e.g., [array([85, 82]), array([80, 51]), array([95, 24]), array([49, 70]), array([ 5, 87])]
    @return the total distance of solution
    '''
    total_distance = 0
    last_customer = 0
    for customer in solution:
        total_distance += euclidean_distance(coordinates[last_customer], coordinates[customer])
        last_customer = customer
    total_distance += euclidean_distance(coordinates[last_customer], coordinates[0])  # Return to the first customer
    return total_distance


def generate_initial_population(num_individuals, num_customers):
    population = []
    for _ in range(num_individuals):
        individual = list(range(num_customers))
        random.shuffle(individual)
        population.append(individual)
    return population


# TODO: implement your mutation operator
def mutate(individual):
    '''
    @describe: the mutation operator
    @param: individual: a list of nodes, e.g., [3, 2, 4, 1]
    @return: individual after mutation
    '''
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]


#TODO: implement your crossover operator
def crossover(parent1, parent2):
    '''
    @describe: the crossover operator
    @param parent1: a list of nodes, e.g., [3, 2, 4, 1]
    @param parent2: a list of nodes, e.g., [4, 3, 2, 1]
    @return child1, child2: two offspring obtained by crossover
    '''
    cut = random.randint(1, len(parent1) - 1)
    child1 = parent1[:cut] + [c for c in parent2 if c not in parent1[:cut]]
    child2 = parent2[:cut] + [c for c in parent1 if c not in parent2[:cut]]
    return child1, child2


# TODO: implement your tournament selection method
def tournament_selection(population, k, coordinates):
    '''
    @describe: the tournament selection method, select k individuals from polulation, the best one is selected as one parent
    @param: Determine by yourself
    @return parent
    '''
    selected = random.sample(population, k)
    return min(selected, key=lambda x: tsp_fitness(x, coordinates))


def tsp_evolutionary_algorithm(num_generations, num_individuals, mutation_rate, num_customers):
    coordinates = generate_tsp_instance(num_customers)
    population = generate_initial_population(num_individuals, num_customers)
    best_solution = min(population, key=lambda x: tsp_fitness(x, coordinates))
    best_fitness = tsp_fitness(best_solution, coordinates)
    print("Best solution of current generation:", best_solution)
    print("Best fitness of current generation:", best_fitness)

    for generation in range(num_generations):
        new_population = []
        for _ in range(num_individuals // 2):
            # Selection
            parent1 = tournament_selection(population, 2, coordinates)
            parent2 = tournament_selection(population, 2, coordinates)

            # Crossover
            child1, child2 = crossover(parent1, parent2)

            # Mutation
            if random.random() < mutation_rate:
                mutate(child1)
            if random.random() < mutation_rate:
                mutate(child2)

            new_population.extend([child1, child2])

        population = new_population
        # Find the best solution in the current population
        best_solution = min(population, key=lambda x: tsp_fitness(x, coordinates))
        best_fitness = tsp_fitness(best_solution, coordinates)
        print("Best solution of current generation:", best_solution)
        print("Best fitness of current generation:", best_fitness)

    # Find the best solution in the final population
    best_solution = min(population, key=lambda x: tsp_fitness(x, coordinates))
    best_fitness = tsp_fitness(best_solution, coordinates)
    return best_solution, best_fitness




if __name__ == "__main__":
    num_generations = 100
    num_individuals = 50
    mutation_rate = 0.2
    num_customers = 20

    best_solution, best_fitness = tsp_evolutionary_algorithm(num_generations, num_individuals, mutation_rate, num_customers)

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)



