import random
import math
import numpy as np
import matplotlib.pyplot as plt
from cec17_functions import cec17_test_func

class DE():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 500
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 50
        self.fun_num = fun_num
        # random.seed(2021+fun_num)
        # np.random.seed(2021+fun_num)

    def fitness_count(self, x):
        f = [0]
        cec17_test_func(x, f, self.dimensions, 1, self.fun_num)
        return f[0]

    def init_pos(self, Recount):
        random.seed(2021+self.fun_num+Recount)
        Pos = np.zeros((self.Search_Agents, self.dimensions))
        for i in range(self.Search_Agents):
            for j in range(self.dimensions):
                Pos[i][j] = random.uniform(self.Lowerbound, self.Uppernound)
        return Pos

    def CheckIndi(self, Indi):
        range_width = self.Uppernound - self.Lowerbound
        for i in range(self.dimensions):
            if Indi[i] > self.Uppernound:
                n = int((Indi[i] - self.Uppernound) / range_width)
                mirrorRange = (Indi[i] - self.Uppernound) - (n * range_width)
                # mirrorRange =1e-4
                Indi[i] = self.Uppernound - mirrorRange
            elif Indi[i] < self.Lowerbound:
                n = int((self.Lowerbound - Indi[i]) / range_width)
                mirrorRange = (self.Lowerbound - Indi[i]) - (n * range_width)
                # mirrorRange =1e-4
                Indi[i] = self.Lowerbound + mirrorRange
            else:
                pass

    def DE_searcher(self, Recount):
        DEer = self.init_pos(Recount)
        Score = float('inf')
        BestDEer = np.zeros((1, self.dimensions))
        DEerFitness = np.zeros(self.Search_Agents)
        Mutation = np.zeros((self.Search_Agents, self.dimensions))
        TrialVector = np.zeros((self.Search_Agents, self.dimensions))
        TrialFitness = np.zeros(self.Search_Agents)
        MAXFEs = 24999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(DEer[r])
            fitness = self.fitness_count(DEer[r])
            DEerFitness[r] = fitness
            FEcount += 1
            if fitness < Score:
                Score = fitness
                BestDEer = DEer[r].copy()
            Convergence[FEcount-1] = Score

        for i in range(1, self.Max_iteration):
            # mutation
            for j in range(0, self.Search_Agents):
                r1 = r2 = r3 = 0
                while r1 == j:
                    r1 = random.randint(0, self.Search_Agents-1)
                while r2 == j or r2 == r1:
                    r2 = random.randint(0, self.Search_Agents-1)
                while r3 == j or r3 == r2 or r3 == r1:
                    r3 = random.randint(0, self.Search_Agents-1)
                Mutation[j] = DEer[r1] + 1.8*(DEer[r2] - DEer[r3])

            # crossover
            for c in range(0, self.Search_Agents):
                select_index = random.randint(0, self.dimensions-1)
                for d in range(self.dimensions):
                    if random.uniform(0, 1) <= 0.9 or select_index == c:
                        TrialVector[c][d] = Mutation[c][d]
                    else:
                        TrialVector[c][d] = DEer[c][d]
                if FEcount > MAXFEs:
                    break
                self.CheckIndi(TrialVector[c])
                TrialFitness[c] = self.fitness_count(TrialVector[c])
                FEcount += 1
                if TrialFitness[c] < Score:
                    Score = TrialFitness[c]
                Convergence[FEcount-1] = Score

            # selection
            for s in range(0, self.Search_Agents):
                if TrialFitness[s] < DEerFitness[s]:
                    DEerFitness[s] = TrialFitness[s]
                    DEer[s] = TrialVector[s]

            for r in range(0, self.Search_Agents):
                if FEcount > MAXFEs:
                    break
                self.CheckIndi(DEer[r])
                fitness = self.fitness_count(DEer[r])
                DEerFitness[r] = fitness
                FEcount += 1
                if fitness < Score:
                    Score = fitness
                    BestDEer = DEer[r].copy()
                Convergence[FEcount-1] = Score
            if FEcount > MAXFEs:
                break
            # print('Iteration:,best score: \n', i, Score)
        return Score, BestDEer, Convergence

    def DE_ReRun(self, Recount):
        MAXFEs = 25000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        for i in range(Recount):
            b_score, b_psoer, score_count = self.DE_searcher(i)
            score_sum += score_count
            total_score[i] = score_count
        avg_scorecount = score_sum/Recount
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./DEResult/50D/Pic/PSOFEs_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score

def main():
    recount = 30
    for nu in range(1, 31):
        if nu != 2:
            DEi = DE(nu)
            c, b = DEi.DE_ReRun(recount)
            np.savetxt('./DEResult/50D/avgResult/avg_score{}.csv'.format(nu), c, delimiter=",")
            np.savetxt('./DEResult/50D/totalResult/totalResult{}.csv'.format(nu), b, delimiter=",")
            # print('s,b', s, b)
            # print("best scorelist:", c)
            print("function:", nu, "is over.")
        # else: continue
    return 0


if __name__=="__main__":
    main()
