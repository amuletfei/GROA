import random
import math
import numpy as np
import matplotlib.pyplot as plt
from cec17_functions import cec17_test_func

class PSO():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 100
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 10
        self.v_min = 0.4
        self.v_max = 0.9
        self.c1 = 2
        self.c2 = 2
        # self.Convergence = np.zeros(self.Max_iteration)
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

    def init_velocity(self):
        V = np.zeros((self.Search_Agents, self.dimensions))
        for i in range(self.Search_Agents):
            for j in range(self.dimensions):
                V[i][j] = random.uniform(self.v_min, self.v_max)
        return V

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

    def check_velocity(self, vel):
        for i in range(self.dimensions):
            if vel[i] > self.v_max:
                vel[i] = self.v_max
            elif vel[i] < self.v_min:
                vel[i] = self.v_min
            else:
                pass

    def PSO_searcher(self, Recount):
        PSOer = self.init_pos(Recount)
        V = self.init_velocity()
        Score = float('inf')
        BestPSOer = np.zeros((1, self.dimensions))
        Pbest = np.zeros((self.Search_Agents, self.dimensions))
        PSOerFitness = np.zeros(self.Search_Agents)
        MAXFEs = 4999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(PSOer[r])
            fitness = self.fitness_count(PSOer[r])
            Pbest[r] = PSOer[r].copy()
            PSOerFitness[r] = fitness
            FEcount += 1
            if fitness < Score:
                Score = fitness
                BestPSOer = PSOer[r].copy()
            Convergence[FEcount-1] = Score

        for i in range(1, self.Max_iteration):
            for j in range(0, self.Search_Agents):
                V[j] = V[j] + self.c1*(Pbest[j]-PSOer[j])*random.random() + \
                       self.c2*(BestPSOer-PSOer[j])*random.random()
                self.check_velocity(V[j])
                PSOer[j] = PSOer[j] + V[j]
            for r in range(0, self.Search_Agents):
                if FEcount > MAXFEs:
                    break
                self.CheckIndi(PSOer[r])
                fitness = self.fitness_count(PSOer[r])
                if fitness < PSOerFitness[r]:
                    Pbest[r] = PSOer[r].copy()
                PSOerFitness[r] = fitness
                FEcount += 1
                if fitness < Score:
                    Score = fitness
                    BestPSOer = PSOer[r].copy()
                Convergence[FEcount-1] = Score
            if FEcount > MAXFEs:
                break
            # print('Iteration:,best score: \n', i, Score)
        return Score, BestPSOer, Convergence

    def PSO_ReRun(self, Recount):
        MAXFEs = 5000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        for i in range(Recount):
            b_score, b_psoer, score_count = self.PSO_searcher(i)
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
        myfig.savefig('./PSOResult/10D/Pic/PSOFEs_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score

def main():
    recount = 30
    for nu in range(1, 31):
        if nu != 2:
            PSOi = PSO(nu)
            c, b = PSOi.PSO_ReRun(recount)
            np.savetxt('./PSOResult/10D/avgResult/avg_score{}.csv'.format(nu), c, delimiter=",")
            np.savetxt('./PSOResult/10D/totalResult/totalResult{}.csv'.format(nu), b, delimiter=",")
            # print('s,b', s, b)
            # print("best scorelist:", c)
            print("function:", nu, "is over.")
        # else: continue
    return 0


if __name__=="__main__":
    main()
