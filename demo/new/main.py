import random
import math
import concurrent.futures

class MessageTask:
    def __init__(self, msgType, usrInst, exeTime, deadLine, idx):
        self.msgType = msgType
        self.usrInst = usrInst
        self.exeTime = exeTime
        self.deadLine = deadLine
        self.idx = idx
        self.endTime = 0

MAX_USER_ID = 10005
ACCURACY = 0.65

def adjust_execution_time(exeTime, accuracy):
    adjusted_time = exeTime * (2 - accuracy * accuracy)
    return round(adjusted_time)

def fitness(assignment, tasks, m, fitness_cache):
    assignment_tuple = tuple(sorted(assignment.items()))
    if assignment_tuple in fitness_cache:
        return fitness_cache[assignment_tuple]

    core_end_times = [0] * m
    affinity_scores = [0] * m
    completed_tasks = [0] * m

    for uid, core_id in assignment.items():
        core = core_id
        if not tasks[uid]:
            continue

        for task in tasks[uid]:
            if core_end_times[core] + task.exeTime <= task.deadLine:
                completed_tasks[core] += 1
                if core_end_times[core] > 0 and core_end_times[core] + task.exeTime <= task.deadLine:
                    affinity_scores[core] += 1
            core_end_times[core] += task.exeTime

    total_affinity_score = sum(affinity_scores)
    total_completed_tasks = sum(completed_tasks)
    total_fitness = total_affinity_score + total_completed_tasks

    fitness_cache[assignment_tuple] = total_fitness
    return total_fitness

def crossover(parent1, parent2):
    child = parent1.copy()
    for uid in parent1.keys():
        if random.random() > 0.5:
            child[uid] = parent2[uid]
    return child

def mutate(assignment, m):
    uid = random.choice(list(assignment.keys()))
    assignment[uid] = random.randint(0, m-1)

def generate_initial_population(tasks, m, population_size):
    population = []
    for _ in range(population_size):
        assignment = {uid: random.randint(0, m-1) for uid in range(MAX_USER_ID) if tasks[uid]}
        population.append(assignment)
    return population

def genetic_algorithm(tasks, m, generations=100, population_size=10, mutation_rate=0.05):
    population = generate_initial_population(tasks, m, population_size)
    fitness_cache = {}

    for generation in range(generations):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            fitness_values = list(executor.map(lambda x: fitness(x, tasks, m, fitness_cache), population))

        population_fitness = sorted(zip(population, fitness_values), key=lambda x: x[1], reverse=True)
        population = [x[0] for x in population_fitness]

        next_generation = population[:5]
        while len(next_generation) < population_size:
            parent1 = random.choice(population[:5])
            parent2 = random.choice(population[:5])
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                mutate(child, m)
            next_generation.append(child)

        population = next_generation

    best_assignment = population[0]
    return best_assignment

def main():
    n, m, c = map(int, input().split())
    tasks = [[] for _ in range(MAX_USER_ID)]
    all_tasks = []
    for idx in range(n):
        msgType, usrInst, exeTime, deadLine = map(int, input().split())
        exeTime = adjust_execution_time(exeTime, ACCURACY)
        deadLine = min(deadLine, c)
        task = MessageTask(msgType, usrInst, exeTime, deadLine, idx)
        tasks[usrInst].append(task)
        all_tasks.append(task)

    for uid in range(MAX_USER_ID):
        tasks[uid].sort(key=lambda x: (x.deadLine, x.exeTime))

    best_assignment = genetic_algorithm(tasks, m)

    cores = [[] for _ in range(m)]
    for uid, core_id in best_assignment.items():
        cores[core_id].extend(tasks[uid])

    output_lines = []
    for coreId, core_tasks in enumerate(cores):
        core_tasks.sort(key=lambda x: x.idx)
        line = f"{len(core_tasks)}"
        for task in core_tasks:
            line += f" {task.msgType} {task.usrInst}"
        output_lines.append(line)

    print('\n'.join(output_lines), end='')

if __name__ == "__main__":
    main()
