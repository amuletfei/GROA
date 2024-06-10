import random
import math
import numpy as np
import matplotlib.pyplot as plt

E = 30000000
G = 12000000
L = 14
tau_max = 13600
sigma_max = 30000
delta_max = 0.25
P = 6000
C1 = 0.10471
C2 = 0.04811
C3 = 1


class WOA():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 400
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 4
        # self.Convergence = np.zeros(self.Max_iteration)
        self.fun_num = fun_num
        # random.seed(2021+fun_num)
        # np.random.seed(2021+fun_num)

    def wbd_object(self, X):
        """
        :param X:
        0.1 <= X[0] <= 2,
        0.1 <= X[1] <= 10,
        0.1 <= X[2] <= 10,
        0.1 <= X[3] <= 2
        :return:
        """
        V_weld = X[0] ** 2 * X[1]
        V_bar = X[2] * X[3] * (L + X[1])
        return (C1 + C3) * V_weld + C2 * V_bar

    def wbd_cons(self, X):
        """
        :return: All cons should be minus than 0
        """
        con1 = tau_max - self.tau(X)
        con2 = delta_max - self.delta(X)
        con3 = X[3] - X[0]
        con4 = 5 - (1 + C1) * X[0] ** 2 + C2 * X[2] * X[3] * (L + X[1])
        con5 = X[0] - 0.125
        con6 = delta_max - self.delta(X)
        con7 = self.Pc(X) - P
        return [-con1, -con2, -con3, -con4, -con5, -con6, -con7]

    def tau(self, X):
        return np.sqrt(self.tau_d(X) ** 2 + 2 * self.tau_d(X) * self.tau_dd(X) * X[1] / (2 * self.R(X)) + self.tau_dd(X) ** 2)

    def tau_d(self, X):
        return P / (np.sqrt(2) * X[0] * X[1])

    def tau_dd(self, X):
        return self.M(X) * self.R(X) / self.J(X)

    def R(self, X):
        return np.sqrt(X[1] ** 2 / 4 + ((X[0] + X[2]) / 2) ** 2)

    def M(self, X):
        return P * (L + X[1] / 2)

    def J(self, X):
        return 2 * (X[0] * X[1] * np.sqrt(2) * (X[1] ** 2 / 12 + ((X[0] + X[2]) / 2) ** 2))

    def sigma(self, X):
        return 6 * P * L / (X[3] * X[2] ** 2)

    def delta(self, X):
        return 4 * P * L ** 3 / (E * X[3] * X[2] ** 2)

    def Pc(self, X):
        coef = 4.013 * E * np.sqrt(X[2] ** 2 * X[3] ** 6 / 36) / (L ** 2)
        return coef * (1 - X[2] / (2 * L) * np.sqrt(E / (4 * G)))

    def fitness_count(self, x, con_conter):
        cons_array = self.wbd_cons(x)
        penalty_function = [1000, 1000, 1000, 1000, 1000, 1000, 1000]
        penalty_term = 0
        for i in range(7):
            if cons_array[i] <= 0:
                con_conter[i] += 1
            else:
                penalty_term += penalty_function[i]
        fitness = self.wbd_object(x) + penalty_term
        return fitness, con_conter

    def init_pos(self, Recount):
        random.seed(2021+self.fun_num+Recount)
        Pos = np.zeros((self.Search_Agents, self.dimensions))
        """
        :param X:
            0.1 <= x[0] <= 2,
            0.1 <= x[1] <= 10,
            0.1 <= x[2] <= 10,
            0.1 <= x[3] <= 2
        :return:
        """
        for i in range(self.Search_Agents):
            Pos[i][0] = random.uniform(0.1, 2)
            Pos[i][1] = random.uniform(0.1, 10)
            Pos[i][2] = random.uniform(0.1, 10)
            Pos[i][3] = random.uniform(0.1, 2)
        return Pos

    def CheckIndi(self, Indi):
        """
        :param X:
            0.1 <= x[0] <= 2,
            0.1 <= x[1] <= 10,
            0.1 <= x[2] <= 10,
            0.1 <= x[3] <= 2
        :return:
        """
        if Indi[0] > 2:
            Indi[0] = 2
        elif Indi[0] < 0.1:
            Indi[0] = 0.1
        else:
            pass
        if Indi[1] > 10:
            Indi[1] = 10
        elif Indi[1] < 0.1:
            Indi[1] = 0.1
        else:
            pass
        if Indi[2] > 10:
            Indi[2] = 10
        elif Indi[2] < 0.1:
            Indi[2] = 0.1
        else:
            pass
        if Indi[3] > 2:
            Indi[3] = 2
        elif Indi[3] < 0.1:
            Indi[3] = 0.1
        else:
            pass

    def WOA_searcher(self, Recount):
        WOAer = self.init_pos(Recount)
        Prevgen = np.zeros((self.Max_iteration, self.Search_Agents, self.dimensions))
        Prevgen[0] = WOAer.copy()
        Score = float('inf')
        BestWOAer = np.zeros(self.dimensions)
        WOAFitness = np.zeros(self.Search_Agents)
        MAXFEs = 19999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)
        con_conter = np.zeros(7)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(WOAer[r])
            fitness, con_conter = self.fitness_count(WOAer[r], con_conter)
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
                fitness, con_conter = self.fitness_count(WOAer[r], con_conter)
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
        return Score, BestWOAer, Convergence, con_conter

    def WOA_ReRun(self, Recount):
        MAXFEs = 20000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        cons_total = np.zeros(7)
        best_solutions = np.zeros((Recount, self.dimensions))
        all_score = np.zeros(Recount)
        for i in range(Recount):
            b_score, b_woaer, score_count, con_c = self.WOA_searcher(i)
            score_sum += score_count
            total_score[i] = score_count
            cons_total += con_c
            best_solutions[i] = b_woaer
            all_score[i] = b_score
        avg_scorecount = score_sum/Recount
        avg_con = cons_total / Recount
        # best_solutions = best_solutions / Recount
        np.savetxt('./WBDResult/con_result/WOAcon_counter_{}.csv'.format('WBD'), avg_con, delimiter=",")
        np.savetxt('./WBDResult/best_solution/WOA_best_solution_{}.csv'.format('WBD'), best_solutions, delimiter=",")
        np.savetxt('./WBDResult/all_score/WOA_all_score_{}.csv'.format('WBD'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./WBDResult/Pic/WOA_WBD_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score

def main():
    recount = 30
    nu = 1
    ROAi = WOA(nu)
    c, b = ROAi.WOA_ReRun(recount)
    np.savetxt('./WBDResult/avg_result/WOA_avg_{}.csv'.format('WBD'), c, delimiter=",")
    np.savetxt('./WBDResult/total_result/WOA_total_{}.csv'.format('WBD'), b, delimiter=",")
    print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
