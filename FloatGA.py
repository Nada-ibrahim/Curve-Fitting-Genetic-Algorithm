
from Parent import Parent
import numpy as np


class FloatGA:
    lb = -10
    ub = 10

    def __init__(self, pop_num, chromosome_length, points_x, points_y):
        self.points_x = points_x
        self.points_y = points_y
        self.chromosome_length = chromosome_length
        self.PC = 0.7
        self.PM = 0.001
        self.generation = self.lb + np.random.random((pop_num, chromosome_length)) * (self.ub - self.lb)
        self.new_generation = np.zeros((pop_num, chromosome_length))
        self.fitness = None
        self.cum_fitness = None
        self.mse = None
        self.pop_num = pop_num
        self.x_array = np.array([[self.points_x[i] ** j for j in range(self.chromosome_length)] for i in range(np.size(self.points_x))])

    def apply_algorithm(self, max_gen):
        for i in range(max_gen):
            self.eval_all_fitness()
            for j in range(int(self.pop_num / 2)):
                parents = self.select_chromosomes()
                self.do_cross_over(parents, j)

            self.do_all_mutation(i, max_gen)
            self.generation = self.new_generation
            self.new_generation = np.zeros((self.pop_num, self.chromosome_length))

        return self.get_optimal_chromosome()

    def eval_all_fitness(self):
        y_predicted = self.generation.dot(self.x_array.transpose())
        self.mse = (1 / np.size(self.points_y)) * np.sum((self.points_y - y_predicted) ** 2, axis=1)
        max_mse = np.max(self.mse)
        self.fitness = max_mse - self.mse
        self.cum_fitness = np.cumsum(self.fitness)

    def select_chromosomes(self):
        p = Parent()
        total_fitness = self.cum_fitness[np.size(self.cum_fitness)-1]
        for i in range(2):
            rand_chrom = np.random.random() * total_fitness
            chromosome = self.generation[self.cum_fitness >= rand_chrom][0]
            p.set_parent(chromosome)
        return p

    def do_cross_over(self, parents, j):
        r = np.random.random()
        L = np.random.randint(1, self.chromosome_length)

        if r <= self.PC:
            offspring1 = np.zeros(self.chromosome_length)
            offspring2 = np.zeros(self.chromosome_length)

            offspring1[0:L] = parents.get_c1()[0:L]
            offspring2[0:L] = parents.get_c2()[0:L]

            offspring1[L:self.chromosome_length] = parents.get_c1()[L:self.chromosome_length]
            offspring2[L:self.chromosome_length] = parents.get_c2()[L:self.chromosome_length]

        else:
            offspring1 = parents.get_c1()
            offspring2 = parents.get_c2()

        self.new_generation[2*j] = offspring1
        self.new_generation[2*j + 1] = offspring2

    def do_all_mutation(self, t, T):
        for i in range(self.new_generation.shape[0]):
            for j in range(self.new_generation.shape[1]):
                lxi = self.new_generation[i,j] - self.lb
                uxi = self.ub - self.new_generation[i,j]
                r2 = np.random.random()
                r = np.random.random()
                b = 1
                if r2 <= self.PM:
                    if r <= 0.5:
                        y = lxi
                        dx = y * (1 - (r ** ((1 - (t / T)) ** b)))
                        self.new_generation[i, j] = self.new_generation[i,j] - dx
                    else:
                        y = uxi
                        dx = y * (1 - (r ** ((1 - (t / T)) ** b)))
                        self.new_generation[i, j] = self.new_generation[i, j] + dx

    def get_optimal_chromosome(self):
        self.eval_all_fitness()
        maxFitness = 0
        optimalChromosome = None
        index = 0
        for i in range(np.size(self.fitness)):
            if self.fitness[i] >= maxFitness:
                maxFitness = self.fitness[i]
                optimalChromosome = self.generation[i]
                index = i

        return optimalChromosome, self.mse[index]
