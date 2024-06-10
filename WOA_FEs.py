import random
import math
import numpy as np
import matplotlib.pyplot as plt
from cec17_functions import cec17_test_func

class WOA():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 100
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 10
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

    def WOA_searcher(self, Recount):
        WOAer = self.init_pos(Recount)
        Prevgen = np.zeros((self.Max_iteration, self.Search_Agents, self.dimensions))
        Prevgen[0] = WOAer.copy()
        Score = float('inf')
        BestWOAer = np.zeros(self.dimensions)
        WOAFitness = np.zeros(self.Search_Agents)
        MAXFEs = 4999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(WOAer[r])
            fitness = self.fitness_count(WOAer[r])
            WOAFitness[r] = fitness
            FEcount += 1
            if (fitness < Score):
                Score = fitness
                BestWOAer = WOAer[r].copy()
            Convergence[FEcount-1] = Score

        for i in range(1, self.Max_iteration):
            for j in range(0, self.Search_Agents):
                a = 2 - 2*(i/self.Max_iteration)
                A = 2*a*random.uniform(0, 1)-a
                C = 2*random.uniform(0, 1)
                p = random.uniform(0, 1)
                if p < 0.5:
                    if abs(A) < 1:
                        D = abs(C*BestWOAer - WOAer[j])
                        WOAer[j] = BestWOAer - A*D
                    else:
                        m = random.randint(0, self.Search_Agents-1)
                        D = abs(C*WOAer[m] - WOAer[j])
                        WOAer[j] = WOAer[m] - A*D
                else:
                    D = abs(BestWOAer - WOAer[j])
                    l = random.uniform(-1, 1)
                    b = 1
                    WOAer[j] = D*np.exp(b*l)*math.cos(2*math.pi*l)+BestWOAer
            for r in range(0, self.Search_Agents):
                if FEcount > MAXFEs:
                    break
                self.CheckIndi(WOAer[r])
                fitness = self.fitness_count(WOAer[r])
                WOAFitness[r] = fitness
                FEcount += 1
                if (fitness < Score):
                    Score = fitness
                    BestWOAer = WOAer[r].copy()
                Convergence[FEcount-1] = Score
            Prevgen[i] = WOAer.copy()
            if FEcount > MAXFEs:
                break
            # print('Iteration:,best score: \n', i, Score)
        return Score, BestWOAer, Convergence

    def WOA_ReRun(self, Recount):
        MAXFEs = 5000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        for i in range(Recount):
            b_score, b_remora, score_count = self.WOA_searcher(i)
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
        myfig.savefig('./WOAResult/10D/Pic/WOAFEs_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score

def main():
    recount = 30
    for nu in range(1, 31):
        if nu != 2:
            ROAi = WOA(nu)
            c, b = ROAi.WOA_ReRun(recount)
            np.savetxt('./WOAResult/10D/avgResult/avg_score{}.csv'.format(nu), c, delimiter=",")
            np.savetxt('./WOAResult/10D/totalResult/totalResult{}.csv'.format(nu), b, delimiter=",")
            # print('s,b', s, b)
            # print("best scorelist:", c)
            print("function:", nu, "is over.")
        # else: continue
    return 0


if __name__=="__main__":
    main()
