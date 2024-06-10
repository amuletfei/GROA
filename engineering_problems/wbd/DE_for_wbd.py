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


class DE():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 400
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 4
        self.fun_num = fun_num
        # random.seed(2021+fun_num)
        # np.random.seed(2021+fun_num)

    # def wbd_cons(self, x):
    #     """
    #     :return: All cons should be minus than 0
    #     σ max = 30000,
    #     P = 6000 lb,
    #     L = 14 in.,
    #     δ max = 0.25 in.,
    #     E = 3×10^6 pis
    #     τ max = 136000 psi
    #     G = 1.2×10^7 psi
    #     """
    #
    #     xichma_max = 30000
    #     p = 6000
    #     l = 14
    #     delta_max = 0.25
    #     e = 3 * 10 ** 6
    #     t_max = 136000
    #     g = 1.2 * 10 ** 7
    #
    #     jj = 2 * np.sqrt(2) * x[0] * x[1] * (x[1] ** 2 / 4 + ((x[0] + x[2]) / 2) ** 2)
    #     R = np.sqrt(x[1] ** 2 / 4 + (x[0] + x[2]) ** 2 / 4)
    #     M = p * (l + x[1] / 2)
    #     tau2 = M * R / jj
    #     tau1 = p / (np.sqrt(2) * x[0] * x[1])
    #     tau = np.sqrt(tau1 ** 2 + 2 * tau1 * tau2 * x[1] / (2 * R) + tau2 ** 2)
    #     xichma_z = 6 * p * l / (e * x[2] ** 2 * x[3])
    #     theta_z = 6 * p * l ** 3 / (e * x[2] ** 2 * x[3])
    #     pc = 4.013 * e * np.sqrt(x[2] ** 2 * x[3] ** 6 / 36) / l ** 2 * (
    #             1 - x[2] * np.sqrt(e / (4 * g)) / (2 * l))
    #
    #     con1 = tau - t_max
    #     con2 = xichma_z - xichma_max
    #     con3 = theta_z - delta_max
    #     con4 = x[0] - x[3]
    #     con5 = p - pc
    #     con6 = 0.125 - x[0]
    #     con7 = 1.10471 * x[0]**2 + 0.04811 * x[2]*x[3]*(14 + x[1]) - 5
    #
    #     return [con1, con2, con3, con4, con5, con6, con7]
    #
    # def wbd_object(self, x):
    #     """
    #     :param x:
    #     0.1 <= x[0] <= 2,
    #     0.1 <= x[2] <= 10,
    #     0.1 <= x[3] <= 10,
    #     0.1 <= x[4] <= 2
    #     :return:
    #     """
    #
    #     return 1.10471 * x[0]**2 * x[1] + 0.04811 * x[2] * x[3] * (14 + x[1])

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
        penalty_function = [10000, 10000, 10000, 10000, 10000, 10000, 10000]
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

    def DE_searcher(self, Recount):
        DEer = self.init_pos(Recount)
        Score = float('inf')
        BestDEer = np.zeros((1, self.dimensions))
        DEerFitness = np.zeros(self.Search_Agents)
        Mutation = np.zeros((self.Search_Agents, self.dimensions))
        TrialVector = np.zeros((self.Search_Agents, self.dimensions))
        TrialFitness = np.zeros(self.Search_Agents)
        MAXFEs = 19999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)
        con_conter = np.zeros(7)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(DEer[r])
            fitness, con_conter = self.fitness_count(DEer[r], con_conter)
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
                Mutation[j] = DEer[r1] + 0.5*(DEer[r2] - DEer[r3])

            # crossover
            for c in range(0, self.Search_Agents):
                select_index = random.randint(0, self.dimensions-1)
                for d in range(self.dimensions):
                    if random.uniform(0, 1) <= 0.8 or select_index == c:
                        TrialVector[c][d] = Mutation[c][d]
                    else:
                        TrialVector[c][d] = DEer[c][d]
                if FEcount > MAXFEs:
                    break
                self.CheckIndi(TrialVector[c])
                TrialFitness[c], con_conter = self.fitness_count(TrialVector[c], con_conter)
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
                fitness, con_conter = self.fitness_count(DEer[r], con_conter)
                DEerFitness[r] = fitness
                FEcount += 1
                if fitness < Score:
                    Score = fitness
                    BestDEer = DEer[r].copy()
                Convergence[FEcount-1] = Score
            if FEcount > MAXFEs:
                break
            # print('Iteration:,best score: \n', i, Score)
        return Score, BestDEer, Convergence, con_conter

    def DE_ReRun(self, Recount):
        MAXFEs = 20000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        cons_total = np.zeros(7)
        best_solutions = np.zeros((Recount, self.dimensions))
        all_score = np.zeros(Recount)
        for i in range(Recount):
            b_score, b_psoer, score_count, con_c = self.DE_searcher(i)
            score_sum += score_count
            total_score[i] = score_count
            cons_total += con_c
            best_solutions[i] = b_psoer
            all_score[i] = b_score
        avg_scorecount = score_sum/Recount
        avg_con = cons_total / Recount
        # best_solutions = best_solutions / Recount
        np.savetxt('./WBDResult/con_result/DEcon_counter_{}.csv'.format('WBD'), avg_con, delimiter=",")
        np.savetxt('./WBDResult/best_solution/DE_best_solution_{}.csv'.format('WBD'), best_solutions, delimiter=",")
        np.savetxt('./WBDResult/all_score/DE_all_score_{}.csv'.format('WBD'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./WBDResult/Pic/DE_WBD_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score


def main():
    recount = 30
    nu = 1
    DEi = DE(nu)
    c, b = DEi.DE_ReRun(recount)
    np.savetxt('./WBDResult/avg_result/DE_avg_{}.csv'.format('WBD'), c, delimiter=",")
    np.savetxt('./WBDResult/total_result/DE_total_{}.csv'.format('WBD'), b, delimiter=",")
    # print('s,b', s, b)
    # print("best scorelist:", c)
    print("function:", nu, "is over.")

    return 0


if __name__=="__main__":
    main()
