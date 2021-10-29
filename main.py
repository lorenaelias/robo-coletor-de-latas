import time
import numpy as np
from random import randint

# SIMULATION
POPULATION_SIZE = 200
NUM_GENERATIONS = 1000
STEPS = 80
CLEANING_SESSIONS = 100
MATRIX_LEN = 10
PERCENTAGE_CANS = 0.2

# DNA
MUTATION = 800
NUM_GENES = 243

# POINTS
REWARD = 10
PICKUP_PENALTY = 1
WALL_PENALTY = 5

def base3_to_base10(base3_str):
    strlen = len(base3_str)
    base10_int = 0
    for index, value in enumerate(base3_str):
        base10_int += int(value) * 3**(strlen-1-index)
    return base10_int

class DNA(object):
    
    def __init__(self, sequence=None):
        if sequence is None:
            self.sequence = np.random.randint(0, 7, NUM_GENES)
        else:
            self.sequence = [self.mutate(x) for x in sequence]

    def get_sequence(self):
        return self.sequence

    def get_gene(self, position):
        return self.sequence[position]

    def crossover(self, dna_p2):

        new1 = [0]*NUM_GENES
        new2 = [0]*NUM_GENES

        a = randint(0, NUM_GENES-1)

        for i in range(0, a):
            new2[i] = self.sequence[i]
            new1[i] = dna_p2.sequence[i]

        for i in range(a, NUM_GENES):
            new1[i] = dna_p2.sequence[i]
            new2[i] = self.sequence[i]

        return DNA(new1), DNA(new2)

    def mutate(self, gene):
        if np.random.randint(1, MUTATION) == 1:
            return (np.random.randint(0, 7))
        else: 
            return gene
class Robot(object):

    def __init__(self, dna=None):
        if dna == None:
            self.dna = DNA()
        else:
            self.dna = dna

        self.fitness = 0
        self.position = {'y': 0, 'x': 0}
        self.moves = {
            0: self.up,
            1: self.right,
            2: self.left,
            3: self.down,
            4: self.random,
            5: self.no_move,
            6: self.pick_can
        }

    def getDNA(self):
        return self.dna

    def getFitness(self):
        return self.fitness
    
    def generateSons(self, p2):
        dna1, dna2 = self.dna.crossover(p2.getDNA())
        return Robot(dna1), Robot(dna2)

    def simulate(self):
        scores = []

        for i in range(0, CLEANING_SESSIONS):
            trialfitness = 0
            matrix = Environment()
            self.position = {'x': 0, 'y': 0}

            for step in range(0, STEPS):
                state = matrix.getState(**self.position)
                gene = self.dna.get_gene(state)
                trialfitness = self.moves[gene](matrix, trialfitness)

            scores.append(trialfitness)

        self.fitness = np.array(scores).mean()
        print("Robot Fitness {}".format(self.fitness))

    def up(self, matrix, fitness):
        if self.position['y'] == 0:
            fitness -= WALL_PENALTY
        else:
            self.position['y'] -= 1
        return fitness

    def right(self, matrix, fitness):
        x, y = matrix.getSize()
        if self.position['x'] == x-1:
            fitness -= WALL_PENALTY
        else:
            self.position['x'] += 1
        return fitness

    def left(self, matrix, fitness):
        if self.position['x'] == 0:
            fitness -= WALL_PENALTY
        else:
            self.position['x'] -= 1
        return fitness
        
    def down(self, matrix, fitness):
        x, y = matrix.getSize()
        if self.position['y'] == y-1:
            fitness -= WALL_PENALTY
        else:
            self.position['y'] += 1
        return fitness

    def random(self, matrix, fitness):
        moves = np.random.choice([self.up, self.right, self.left, self.down])
        return moves(matrix, fitness)

    def no_move(self, matrix, fitness):
        return fitness

    def pick_can(self, matrix, fitness):
        if matrix.removeCan(**self.position):
            fitness += REWARD
        else:
            fitness -= PICKUP_PENALTY
        return fitness

class Environment(object):
    
    def __init__(self):
        self.matrix = np.rint(np.random.rand(MATRIX_LEN, MATRIX_LEN)).astype(np.int64)

    def getState(self, x, y):
        state = [
            str(self.position_state(x, y-1)),
            str(self.position_state(x+1, y)),
            str(self.position_state(x-1, y)),
            str(self.position_state(x, y+1)),
            str(self.position_state(x, y))
        ]
        return base3_to_base10(''.join(state))
    
    def getSize(self):
        return self.matrix.shape

    def position_state(self, x, y):
        try:
            if x < 0 or y < 0:
                raise Exception
            return self.matrix[x, y]
        except:
            return 2

    def removeCan(self, x, y):
        if self.matrix[x, y]:
            self.matrix[x, y] = 0
            return True
        else:
            return False

def normalize(x, minf, maxf):
    return (x - minf) / (maxf - minf)

def father_probability(population):
    popfitness = [robot.getFitness() for robot in population]
    minfitness = min(popfitness)
    maxfitness = max(popfitness)
    normalized = list(map(lambda x: normalize(x, minfitness, maxfitness), popfitness))
    total = sum(normalized)
    return list(map(lambda x: x/total, normalized))

def choose_parents(population):
    p1, p2 = randint(0,POPULATION_SIZE-1), randint(0,POPULATION_SIZE-1)
    while(p1 == p2):
        p1,p2 = randint(0,POPULATION_SIZE-1), randint(0,POPULATION_SIZE-1)
    return population[p1], population[p2]

def execution():

    population = np.array([Robot() for i in range(0, POPULATION_SIZE)])

    for gen in range(0, NUM_GENERATIONS):

        for robot in population:
            robot.simulate()

        gbest = max([robot.getFitness() for robot in population])

        print("Generation {}: {}".format(gen, gbest))
        new_population = list()

        while len(new_population) < POPULATION_SIZE:
            p1, p2 = np.random.choice(population, size=2, p=father_probability(population))
            # p1, p2 = choose_parents(population)
            new1, new2 = p1.generateSons(p2)
            new_population.append(new1)
            new_population.append(new2)

        population = new_population

    fittestRobot = getFittest(population)
    return fittestRobot, gbest

def getFittest(population):
    fittest = None
    for robot in population:
        if fittest is None:
            fittest = robot
        else:
            if fittest.getFitness() < robot.getFitness():
                fittest = robot
    return fittest

if __name__=='__main__':

    mean_time = 0
    worst = 100000
    best = -100000
    all_results = 0

    for i in range(10):
        initial_time = time.time()
        fittestRobot, gbest = execution()
        total_time = time.time() - initial_time
        mean_time += total_time

        if gbest > best:
            best = gbest
            best_robot = fittestRobot
        if gbest < worst:
            worst = gbest
            worst_robot = fittestRobot
        
        all_results += gbest

        print("L: {}".format(STEPS))

        f = open("results.txt", "a")
        f.write(f'Execução ${i}---------------------------------------\n')
        f.write(f'Melhor: {best}\n\n')
        f.write(f'Melhor robô: {fittestRobot.getDNA().get_sequence()}\n\n')
        f.write(f'Pior: {worst}\n\n')
        f.write(f'Tempo: {total_time}\n\n')
        f.close()

    f = open("results.txt", "a")
    f.write(f'FINAL ---------------------------------------\n')
    f.write(f'Melhor: {best}\n\n')
    f.write(f'Melhor: {best_robot.getDNA().get_sequence()}\n\n')
    f.write(f'Pior: {worst}\n\n')
    f.write(f'Media: {all_results/10}\n\n')
    f.write(f'Tempo Médio: {mean_time/10}\n\n')
    f.close()

    print(''.join([str(int(x)) for x in fittestRobot.getDNA().get_sequence()]))