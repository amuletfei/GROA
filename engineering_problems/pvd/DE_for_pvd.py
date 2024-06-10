import random
import math
import numpy as np
import matplotlib.pyplot as plt


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

    def PVD_obj(self, X):
        """
        :param X:
        0 <= X[0] <= 99,
        0 <= X[1] <= 99,
        10 <= X[2] <= 200,
        10 <= X[3] <= 200
        :return:
        """
        return 0.6224 * X[0] * X[2] * X[3] + 1.7781 * X[1] * X[2] ** 2 + 3.1661 * X[0] ** 2 * X[3] + 19.84 * X[0] ** 2 * \
               X[2]

    def PVD_cons(self, X):
        """
        :return: All cons should be minus than 0
        """
        con1 = -X[0] + 0.0193 * X[2]
        con2 = -X[1] + 0.00954 * X[2]
        con3 = -np.pi * X[2] ** 2 * X[3] - 4 / 3 * np.pi * X[2] ** 3 + 1296000
        con4 = X[3] - 240
        return [con1, con2, con3, con4]

    def fitness_count(self, x, con_conter):
        cons_array = self.PVD_cons(x)
        penalty_function = [1000000, 1000000, 1000000, 1000000]  # [1000000, 1000000, 1000000, 1000000]
        penalty_term = 0
        for i in range(4):
            if cons_array[i] <= 0:
                con_conter[i] += 1
            else:
                penalty_term += penalty_function[i]
        fitness = self.PVD_obj(x) + penalty_term
        return fitness, con_conter

    def init_pos(self, Recount):
        random.seed(2021+self.fun_num+Recount)
        Pos = np.zeros((self.Search_Agents, self.dimensions))
        """
           :param X:
           0 <= X[0] <= 99,
           0 <= X[1] <= 99,
           10 <= X[2] <= 200,
           10 <= X[3] <= 200
           :return:
           """
        for i in range(self.Search_Agents):
            Pos[i][0] = random.uniform(0, 99)
            Pos[i][1] = random.uniform(0, 99)
            Pos[i][2] = random.uniform(10, 200)
            Pos[i][3] = random.uniform(10, 200)
        return Pos

    def CheckIndi(self, Indi):
        """
              :param X:
              0 <= X[0] <= 99,
              0 <= X[1] <= 99,
              10 <= X[2] <= 200,
              10 <= X[3] <= 200
              :return:
              """
        if Indi[0] > 99:
            Indi[0] = 99
        elif Indi[0] < 0:
            Indi[0] = 0
        else:
            pass
        if Indi[1] > 99:
            Indi[1] = 99
        elif Indi[1] < 0:
            Indi[1] = 0
        else:
            pass
        if Indi[2] > 200:
            Indi[2] = 200
        elif Indi[2] < 10:
            Indi[2] = 10
        else:
            pass
        if Indi[3] > 200:
            Indi[3] = 200
        elif Indi[3] < 10:
            Indi[3] = 10
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
        con_conter = np.zeros(4)

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
        cons_total = np.zeros(4)
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
        np.savetxt('./PVDResult/con_result/DEcon_counter_{}.csv'.format('PVD'), avg_con, delimiter=",")
        np.savetxt('./PVDResult/best_solution/DE_best_solution_{}.csv'.format('PVD'), best_solutions, delimiter=",")
        np.savetxt('./PVDResult/all_score/DE_all_score_{}.csv'.format('PVD'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./PVDResult/Pic/DE_PVD_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')
        return avg_scorecount, total_score


def main():
    recount = 30
    nu = 1
    DEi = DE(nu)
    c, b = DEi.DE_ReRun(recount)
    np.savetxt('./PVDResult/avg_result/DE_avg_{}.csv'.format('PVD'), c, delimiter=",")
    np.savetxt('./PVDResult/total_result/DE_total_{}.csv'.format('PVD'), b, delimiter=",")
    # print('s,b', s, b)
    # print("best scorelist:", c)
    print("function:", nu, "is over.")

    return 0


if __name__=="__main__":
    main()
