import random
import math
import numpy as np
import matplotlib.pyplot as plt


class PSO():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 400
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 11
        self.v_min = -2
        self.v_max = 2
        self.c1 = -1
        self.c2 = 1
        # self.Convergence = np.zeros(self.Max_iteration)
        self.fun_num = fun_num
        # random.seed(2021+fun_num)
        # np.random.seed(2021+fun_num)

    def csid_cons(self, x):
        """
        :return: All cons should be minus than 0
        """
        con1 = 1.16 - 0.3717 * x[1] * x[3] - 0.00931 * x[1] * x[9] - 0.484 * x[2] * x[8] + 0.01343 * x[5] * x[9] - 1
        con2 = 46.36 - 9.9 * x[1] - 12.9 * x[0] * x[1] + 0.1107 * x[2] * x[9] - 32
        con3 = 33.86 + 2.95 * x[2] + 0.1792 * x[2] - 5.057 * x[0] * x[1] - 11.0 * x[1] * x[7] - 0.0215 * x[4] * x[
            9] - 9.98 * x[6] * x[7] + 22.0 * x[7] * x[8] - 32
        con4 = 28.98 + 3.818 * x[2] - 4.2 * x[0] * x[1] + 0.0207 * x[4] * x[9] + 6.63 * x[5] * x[8] - 7.7 * x[6] * x[
            7] + 0.32 * x[8] * x[9] - 32
        con5 = 0.261 - 0.0159 * x[0] * x[1] - 0.188 * x[0] * x[7] - 0.019 * x[1] * x[6] + 0.0144 * x[2] * x[
            4] + 0.0008757 * x[4] * x[9] + 0.08045 * x[5] * x[8] + 0.00139 * x[7] * x[10] + 0.00001575 * x[9] * x[
                   10] - 0.32
        con6 = 0.214 + 0.00817 * x[4] - 0.131 * x[0] * x[7] - 0.0704 * x[0] * x[8] + 0.03099 * x[1] * x[5] - 0.018 * x[
            1] * x[6] + 0.0208 * x[2] * x[7] + 0.121 * x[2] * x[8] - 0.00364 * x[4] * x[5] + 0.0007715 * x[4] * x[
                   9] - 0.0005354 * x[5] * x[9] + 0.00121 * x[7] * x[10] + 0.00184 * x[8] * x[9] - 0.02 * x[
                   1] ** 2 - 0.32
        con7 = 0.74 - 0.61 * x[1] - 0.613 * x[2] * x[7] + 0.001232 * x[2] * x[9] - 0.166 * x[6] * x[8] + 0.227 * x[
            1] ** 2 - 0.32
        con8 = 4.72 - 0.5 * x[3] - 0.19 * x[1] * x[2] - 0.0122 * x[3] * x[9] * 0.009325 * x[5] * x[9] + 0.000191 * x[
            10] ** 2 - 4
        con9 = 10.58 - 0.674 * x[0] * x[1] - 1.95 * x[1] * x[7] + 0.02054 * x[2] * x[9] - 0.0198 * x[3] * x[9] + 0.028 * \
               x[5] * x[9] - 9.9
        con10 = 16.45 - 0.489 * x[2] * x[6] - 0.843 * x[2] * x[6] + 0.0432 * x[8] * x[9] - 0.0556 * x[8] * x[
            10] - 0.000786 * x[10] ** 2 - 15.7
        return [con1, con2, con3, con4, con5, con6, con7, con8, con9, con10]

    def csid_obj(self, x):
        """
        :param x:
        0.5 <= X[0],x[1],x[2],x[3],x[4],x[5],x[6] <= 1.5,
        0.192 <= X[7],x[8] <= 0.345,
        -30 <= X[9],x[10] <= 30,
        """
        return 1.98 + 4.90 * x[0] + 6.67 * x[1] + 6.98 * x[2] + 4.01 * x[3] + 1.78 * x[4] + 2.73 * x[6]

    def fitness_count(self, x, con_conter):
        cons_array = self.csid_cons(x)
        penalty_function = [10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000]
        penalty_term = 0
        for i in range(10):
            if cons_array[i] <= 0:
                con_conter[i] += 1
            else:
                penalty_term += penalty_function[i]
        fitness = self.csid_obj(x) + penalty_term
        return fitness, con_conter

    def init_pos(self, Recount):
        random.seed(2021 + self.fun_num + Recount)
        Pos = np.zeros((self.Search_Agents, self.dimensions))
        """
        :param x:
        0.5 <= X[0],x[1],x[2],x[3],x[4],x[5],x[6] <= 1.5,
        0.192 <= X[7],x[8] <= 0.345,
        -30 <= X[9],x[10] <= 30,
        """
        for i in range(self.Search_Agents):
            Pos[i][0] = random.uniform(0.5, 1.5)
            Pos[i][1] = random.uniform(0.5, 1.5)
            Pos[i][2] = random.uniform(0.5, 1.5)
            Pos[i][3] = random.uniform(0.5, 1.5)
            Pos[i][4] = random.uniform(0.5, 1.5)
            Pos[i][5] = random.uniform(0.5, 1.5)
            Pos[i][6] = random.uniform(0.5, 1.5)
            Pos[i][7] = random.uniform(0.192, 0.345)
            Pos[i][8] = random.uniform(0.192, 0.345)
            Pos[i][9] = random.uniform(-30, 30)
            Pos[i][10] = random.uniform(-30, 30)

        return Pos

    def CheckIndi(self, Indi):
        """
        :param x:
        0.5 <= X[0],x[1],x[2],x[3],x[4],x[5],x[6] <= 1.5,
        0.192 <= X[7],x[8] <= 0.345,
        -30 <= X[9],x[10] <= 30,
        """
        if Indi[0] > 1.5:
            Indi[0] = 1.5
        elif Indi[0] < 0.5:
            Indi[0] = 0.5
        else:
            pass
        if Indi[1] > 1.5:
            Indi[1] = 1.5
        elif Indi[1] < 0.5:
            Indi[1] = 0.5
        else:
            pass
        if Indi[2] > 1.5:
            Indi[2] = 1.5
        elif Indi[2] < 0.5:
            Indi[2] = 0.5
        if Indi[3] > 1.5:
            Indi[3] = 1.5
        elif Indi[3] < 0.5:
            Indi[3] = 0.5
        if Indi[4] > 1.5:
            Indi[4] = 1.5
        elif Indi[4] < 0.5:
            Indi[4] = 0.5
        if Indi[5] > 1.5:
            Indi[5] = 1.5
        elif Indi[5] < 0.5:
            Indi[5] = 0.5
        if Indi[6] > 1.5:
            Indi[6] = 1.5
        elif Indi[6] < 0.5:
            Indi[6] = 0.5
        if Indi[7] > 0.345:
            Indi[7] = 0.345
        elif Indi[7] < 0.192:
            Indi[7] = 0.192
        if Indi[8] > 0.345:
            Indi[8] = 0.345
        elif Indi[8] < 0.192:
            Indi[8] = 0.192
        if Indi[9] > 30:
            Indi[9] = 30
        elif Indi[9] < -30:
            Indi[9] = -30
        if Indi[10] > 30:
            Indi[10] = 30
        elif Indi[10] < -30:
            Indi[10] = -30
        else:
            pass

    def init_velocity(self):
        V = np.zeros((self.Search_Agents, self.dimensions))
        for i in range(self.Search_Agents):
            for j in range(self.dimensions):
                V[i][j] = random.uniform(self.v_min, self.v_max)
        return V

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
        MAXFEs = 19999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)
        con_conter = np.zeros(10)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(PSOer[r])
            fitness, con_conter = self.fitness_count(PSOer[r], con_conter)
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
                fitness, con_conter = self.fitness_count(PSOer[r], con_conter)
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
        return Score, BestPSOer, Convergence, con_conter

    def PSO_ReRun(self, Recount):
        MAXFEs = 20000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        cons_total = np.zeros(10)
        best_solutions = np.zeros((Recount, self.dimensions))
        all_score = np.zeros(Recount)
        for i in range(Recount):
            b_score, b_psoer, score_count, con_c = self.PSO_searcher(i)
            score_sum += score_count
            total_score[i] = score_count
            cons_total += con_c
            best_solutions[i] = b_psoer
            all_score[i] = b_score
        avg_scorecount = score_sum/Recount
        avg_con = cons_total / Recount
        # best_solutions = best_solutions / Recount
        np.savetxt('./CSIDResult/con_result/PSOcon_counter_{}.csv'.format('CSID'), avg_con, delimiter=",")
        np.savetxt('./CSIDResult/best_solution/PSO_best_solution_{}.csv'.format('CSID'), best_solutions, delimiter=",")
        np.savetxt('./CSIDResult/all_score/PSO_all_score_{}.csv'.format('CSID'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./CSIDResult/Pic/PSO_CSID_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score


def main():
    recount = 30
    nu = 1
    PSOi = PSO(nu)
    c, b = PSOi.PSO_ReRun(recount)
    np.savetxt('./CSIDResult/avg_result/PSO_avg_{}.csv'.format('CSID'), c, delimiter=",")
    np.savetxt('./CSIDResult/total_result/PSO_total_{}.csv'.format('CSID'), b, delimiter=",")
    # print('s,b', s, b)
    # print("best scorelist:", c)
    print("function:", nu, "is over.")
    # else: continue
    return 0


if __name__=="__main__":
    main()
