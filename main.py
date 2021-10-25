import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations

class CanPickerRobot:

    population = []
    matrix = []
    cities = []

    def __init__(self, matrixSize, populationSize, numGenerations, mutProb, crossProb, cellTrashProbab):
        
        self.matrixSize = matrixSize
        self.populationSize = populationSize
        self.numGenerations = numGenerations
        self.mutationProb = mutProb
        self.crossoverProb = crossProb
        self.cellTrashProbab = cellTrashProbab

        self.createMatrixWithCan()

        # Cria a população inicial
        # for i in range(populationSize):
        #     self.population.append( self.createElement() ) 

    # Função que cria a malha nxn incluindo as latas
    def createMatrixWithCan(self):
        canPositioningPossibilities = ['o','x']
        self.matrix = np.random.choice(canPositioningPossibilities, size=(self.matrixSize, self.matrixSize), p=(1-self.cellTrashProbab, self.cellTrashProbab))
        self.matrix[:,[0,self.matrixSize-1]] = 'w'
        self.matrix[[0,self.matrixSize-1], :] = 'w'
        self.show_matrix(self.matrix)
    
    # Função que mostra a malha
    def show_matrix(self, element):
        print(element)

    # Função de crossover pmx
    def pmx_cx(self, p1, p2):

        a = random.randint(0,self.matrixSize-2)
        b = random.randint(a+1,self.matrixSize-1)

        f1, f2 = [0]*self.matrixSize, [0]*self.matrixSize

        for i in range(len(p1)):
            f1[i] = p1[i]
        for i in range(len(p2)):
            f2[i] = p2[i]

        numCities = len(f1)
        idx1 = [0] * numCities
        idx2 = [0] * numCities

        for i, x in enumerate(f1):
            idx1[x] = i
        for i, x in enumerate(f2):
            idx2[x] = i

        for i in range(a, b+1) :
            f1[i], f2[i] = f2[i], f1[i]

        irange = list(range(0,a)) + list(range(b+1, numCities))

        for i in irange:
            x = f1[i]
            while idx2[x] >=a and idx2[x] <= b :
                x = f2[idx2[x]]
            f1[i] = x

            x = f2[i]
            while idx1[x] >= a and idx1[x] <= b:
                x = f1[idx1[x]]
            f2[i] = x

        return f1, f2

    def handle_conversion(self):
        occ = self.population.count(self.population[0])
        if (occ/self.populationSize >= 0.90):
            return True
        else:
            return False

    # Função principal do programa
    def main(self, iteration):
        f = open("results.txt","a")
        f.write(f'Execução {iteration}:\n')
        self.simulate()
        self.show_matrix(self.population[0])
        f.write(f'Melhor = {self.evaluateElement(self.population[0])}\n')
        f.write(f'Pior = {self.evaluateElement(self.population[self.populationSize-1])}\n')
        f.write(f'Media = {self.calculateMean()}\n')
        f.write("\n")
        f.close()
        
    def calculateMean(self):
        s = 0.0
        for i in self.population:
            s+=self.evaluateElement(i)
        return s/self.populationSize

    # Função da roleta que escolhe os pais para a reprodução
    def roullete(self):
        p1, p2 = random.randint(0,self.populationSize-1),random.randint(0,self.populationSize-1)
        while(p1 == p2):
            p1,p2 = random.randint(0,self.populationSize-1),random.randint(0,self.populationSize-1)
        return self.population[p1],self.population[p2]

    # Função de mutação
    def mutate(self,elem):

        x = random.uniform(0, 1)
            
        if(x<=self.mutationProb):
            a = random.randint(0,self.matrixSize-1)
            b = random.randint(0,self.matrixSize-1)

            while(a==b):
                b = random.randint(0,self.matrixSize-1)
                
            aux = elem[a]
            elem[a] = elem[b]
            elem[b] = aux

    # Simula a evolução, ou seja, o passar das gerações com a evolução da população
    def simulate(self):

        for gen  in range(self.numGenerations):
            print(f'Generation {gen}:\n')
            childreen = []
            
            for j in range(int((self.crossoverProb*self.populationSize)/2)):
                p1,p2 = self.roullete()
                f1,f2 = self.pmx_cx(p1, p2)(p1, p2)
                
                self.mutate(f1)
                self.mutate(f2)

                childreen.append(f1)
                childreen.append(f2)
            
            for f in childreen:
                self.population.append(f)
            
            self.population.sort(key=self.evaluateElement)

            while(len(self.population)>self.populationSize):
                self.population.pop()
            
            if(self.handle_conversion()):
                break

# params: matrixSize, populationSize, numGenerations, mutProb, crossProb, cellTrashProb
canPickerRobot = CanPickerRobot(8, 150, 500, 0.1, 0.8, 0.2)

# for i in range(10):
#     canPickerRobot.main(i)