import random
import math
import numpy as np
import matplotlib.pyplot as plt
from cec17_functions import cec17_test_func

class ROA():
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

    def ROA_searcher(self, Recount):
        Remora = self.init_pos(Recount)
        Prevgen = np.zeros((self.Max_iteration, self.Search_Agents, self.dimensions))
        Prevgen[0] = Remora.copy()
        Score = float('inf')
        BestRemora = np.zeros((1, self.dimensions))
        RemoraFitness = np.zeros(self.Search_Agents)
        MAXFEs = 4999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(Remora[r])
            fitness = self.fitness_count(Remora[r])
            RemoraFitness[r] = fitness
            FEcount += 1
            if (fitness < Score):
                Score = fitness
                BestRemora = Remora[r].copy()
            Convergence[FEcount-1] = Score

        for i in range(1, self.Max_iteration):
            if i < 1:
                PreR = Prevgen[0, :]
            else:
                PreR = Prevgen[i-1, :]
            for j in range(0, self.Search_Agents):
                RemoraAtt = Remora[j]+(Remora[j]-PreR[j])*random.random()   #equation 2  正确应为（-1，1）
                self.CheckIndi(RemoraAtt)  # 2023-1-20
                fitnessAtt = self.fitness_count(RemoraAtt)
                fitnessI = RemoraFitness[j]
                if fitnessI > fitnessAtt:
                    r_best = np.argmin(RemoraFitness, axis=0)
                    current_best = Remora[r_best].copy()
                    v = 2*(1-i/self.Max_iteration)              #equation 12
                    b = 2*v*random.random()-v                   #equation 11
                    c = 0.1
                    a = b*(Remora[j, :]-c*current_best)           #equation 10
                    Remora[j, :] = Remora[j, :]+a               #equation 9
                elif random.randint(0, 1) == 0:
                    r_best = np.argmin(RemoraFitness, axis=0)
                    current_best = Remora[r_best].copy()
                    a = -(1+i/self.Max_iteration)               #equation 7
                    alpha = random.random()*(a-1)+1             #equation 6
                    d = abs(current_best-Remora[j, :])            #equation 8
                    Remora[j, :] = d*np.exp(alpha)*math.cos(2*math.pi*alpha)+Remora[j, :]     #equation 5
                else:
                    m = random.randint(0, self.Search_Agents-1)
                    Remora[j, :] = BestRemora-(random.random()*((BestRemora+Remora[m, :])/2)-Remora[m, :]) #equation 1
            for r in range(0, self.Search_Agents):
                if FEcount > MAXFEs:
                    break
                self.CheckIndi(Remora[r])
                fitness = self.fitness_count(Remora[r])
                RemoraFitness[r] = fitness
                FEcount += 1
                if (fitness < Score):
                    Score = fitness
                    BestRemora = Remora[r].copy()
                Convergence[FEcount-1] = Score
            Prevgen[i] = Remora.copy()
            if FEcount > MAXFEs:
                break
            # print('Iteration:,best score: \n', i, Score)
        return Score, BestRemora, Convergence

    def ROA_ReRun(self, Recount):
        MAXFEs = 5000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        for i in range(Recount):
            b_score, b_remora, score_count = self.ROA_searcher(i)
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
        myfig.savefig('./ROAFEsResult/10D/Pic/ROAFEs_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score

def main():
    recount = 30
    for nu in range(1, 31):
        if nu != 2:
            ROAi = ROA(nu)
            c, b = ROAi.ROA_ReRun(recount)
            np.savetxt('./ROAFEsResult/10D/avgResult/avg_score{}.csv'.format(nu), c, delimiter=",")
            np.savetxt('./ROAFEsResult/10D/totalResult/totalResult{}.csv'.format(nu), b, delimiter=",")
            # print('s,b', s, b)
            # print("best scorelist:", c)
            print("function:", nu, "is over.")
        # else: continue
    return 0


if __name__=="__main__":
    main()
