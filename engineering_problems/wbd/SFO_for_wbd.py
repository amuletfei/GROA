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


class MSFO():
    def __init__(self,  fun_num):
        self.Sailfish_pop = 50
        self.Sardine_pop = 50
        self.Max_iteration = 400
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 4
        self.EPSILON = 0.001
        self.fun_num = fun_num
        # self.Convergence = np.zeros(self.Max_iteration+1)

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

    def init_pos(self, search_agents, dimension, upperbound, lowerbound, Recount):
        random.seed(2021+self.fun_num+Recount)
        Pos = np.zeros((search_agents, dimension))
        """
        :param X:
            0.1 <= x[0] <= 2,
            0.1 <= x[1] <= 10,
            0.1 <= x[2] <= 10,
            0.1 <= x[3] <= 2
        :return:
        """
        for i in range(search_agents):
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

    def Sailfish_Update(self, OneSailfish, BestSailfish, BestSardine, SailfihsNum, SardineNum):
        PD = 1 - SailfihsNum/(SailfihsNum + SardineNum)
        la = 2 * random.random()*PD - PD
        OneSailfish = BestSailfish - la * (random.random()*0.5*(BestSailfish + BestSardine) - OneSailfish)  # can use bestfihs

        return OneSailfish

    def Sartine_Update(self, OneSardine, BestSailfish, itr):
        A = 4 * (1.0 - itr/self.Max_iteration)
        AP = A * (1 - (2 * itr * self.EPSILON))
        r = random.random()
        OneSardine = r * (BestSailfish - OneSardine + AP)

        return OneSardine

    def Switch_Update(self, sailfish, sardine, sailfishfitness, sartinefitness):
        selectsail = random.randint(0, self.Sailfish_pop - 1)
        if sailfishfitness[selectsail] > min(sartinefitness):
            swsardine = np.argmin(sartinefitness)
            sailfish[selectsail] = sardine[swsardine]
            sailfishfitness[selectsail] = sartinefitness[swsardine]
            sardine = np.delete(sardine, swsardine, axis=0)
            sartinefitness = np.delete(sartinefitness, swsardine)
            self.Sardine_pop = self.Sardine_pop - 1
        return sailfish, sardine, sailfishfitness, sartinefitness

    def MSFO_Searcher(self, Recount):
        Sailfish = self.init_pos(self.Sailfish_pop, self.dimensions, self.Uppernound, self.Lowerbound, Recount)
        Sardine = self.init_pos(self.Sardine_pop, self.dimensions, self.Uppernound, self.Lowerbound, Recount)
        Score = float('inf')
        SailfishScore = float('inf')
        SardineScore = float('inf')
        BestSailfish = np.zeros(self.dimensions)
        BestSardine = np.zeros(self.dimensions)
        BestFish = np.zeros(self.dimensions)
        SailfishFitness = np.zeros(self.Sailfish_pop)
        SardineFitness = np.zeros(self.Sardine_pop)
        PreSailfishFitness = np.zeros(self.Sailfish_pop)
        PreSardineFitness = np.zeros(self.Sardine_pop)
        FishType = np.zeros(self.Sailfish_pop+self.Sardine_pop)
        SailfishNumRecorder = np.zeros((self.Max_iteration, 1))
        MutationLevel = np.zeros(self.Sailfish_pop+self.Sardine_pop)
        PreSailfishFitness = PreSailfishFitness + Score
        PreSardineFitness = PreSardineFitness + Score
        MAXFEs = 19999
        # self.Convergence = np.zeros(self.Max_iteration+1)
        Convergence = np.zeros(MAXFEs+1)
        FEcount = 0
        con_conter = np.zeros(7)

        for i in range(self.Sailfish_pop):
            self.CheckIndi(Sailfish[i])
            fitness, con_conter = self.fitness_count(Sailfish[i], con_conter)
            FEcount += 1
            SailfishFitness[i] = fitness
            if fitness < Score:
                Score = fitness
                BestFish = Sailfish[i].copy()
            if fitness < SailfishScore:
                SailfishScore = fitness
                BestSailfish = Sailfish[i].copy()
            Convergence[FEcount-1] = Score

        for i in range(self.Sardine_pop):
            self.CheckIndi(Sardine[i])
            fitness, con_conter = self.fitness_count(Sardine[i], con_conter)
            FEcount += 1
            SardineFitness[i] = fitness
            # if fitness < Score:
            #     Score = fitness
            #     BestFish = Sardine[i].copy()
            if fitness < SardineScore:
                SardineScore = fitness
                BestSardine = Sardine[i].copy()
            Convergence[FEcount-1] = Score

        # Convergence[0] = Score
        # apc = 0
        for t in range(self.Max_iteration):
            sailfishnum = self.Sailfish_pop
            sardinenum = self.Sardine_pop
            SailfishNumRecorder[t] = self.Sardine_pop
            # PD = 1 - sailfishnum/(sailfishnum + sardinenum)
            # la = 2 * random.random()*PD - PD

            for p in range(self.Sailfish_pop):
                Sailfish[p] = self.Sailfish_Update(Sailfish[p], BestSailfish, BestSardine, sailfishnum, sardinenum)

            A = 4 * (1.0 - t/self.Max_iteration)
            AP = A * (1 - (2 * t * self.EPSILON))
            if AP > 0.5:
                # print('ap>:', apc)
                # apc += 1
                for s in range(self.Sardine_pop):
                    Sardine[s] = self.Sartine_Update(Sardine[s], BestSailfish, t)
            else:
                updatenum = round(sardinenum * AP)
                updatedim = round(self.dimensions * AP)
                temparg = np.arange(0, self.Sardine_pop, 1)
                update_arg = np.zeros(updatenum)
                tempdim = np.arange(0, self.dimensions, 1)
                update_dim = np.zeros(updatedim)

                for i in range(updatenum):
                    argt = random.randint(0, len(temparg)-1)
                    update_arg[i] = temparg[argt]
                    temparg = np.delete(temparg, argt)
                for i in range(updatedim):
                    argd = random.randint(0, len(tempdim)-1)
                    update_dim[i] = tempdim[argd]
                    tempdim = np.delete(tempdim, argd)

                # print('uparg:', updatenum, update_arg)
                # print('updim:', updatedim, update_dim)
                for u in range(updatenum):
                    r = random.random()
                    ar = int(update_arg[u])
                    # print('ar', ar)
                    for ud in range(updatedim):
                        # print('sarup', Sardine[ar][ud])
                        d = int(update_dim[ud])
                        Sardine[ar][d] = r * (BestSailfish[d] - Sardine[ar][d] + AP)

            for j in range(self.Sailfish_pop):
                if FEcount > MAXFEs:
                    # print('itr', t)
                    # print('FEc', FEcount)
                    # print('lencon', len(Convergence))
                    break
                self.CheckIndi(Sailfish[j])
                fitness, con_conter = self.fitness_count(Sailfish[j], con_conter)
                FEcount += 1
                if fitness < SailfishScore:
                    SailfishScore = fitness
                    BestSailfish = Sailfish[j].copy()
                if fitness < Score:
                    Score = fitness
                    BestFish = Sailfish[j].copy()
                PreSailfishFitness[j] = SailfishFitness[j]
                SailfishFitness[j] = fitness
                Convergence[FEcount-1] = Score
            if FEcount > MAXFEs:
                # print('itr', t)
                # print('FEc', FEcount)
                # print('lencon', len(Convergence))
                break
            for k in range(self.Sardine_pop):
                if FEcount > MAXFEs:
                    # print('itr', t)
                    # print('FEc', FEcount)
                    # print('lencon', len(Convergence))
                    break
                self.CheckIndi(Sardine[k])
                fitness, con_conter = self.fitness_count(Sardine[k], con_conter)
                FEcount += 1
                if fitness < SardineScore:
                    SardineScore = fitness
                    BestSardine = Sardine[k].copy()
                # if fitness < Score:
                #     Score = fitness
                #     BestFish = Sardine[k].copy()
                PreSardineFitness[k] = SardineFitness[k]
                SardineFitness[k] = fitness
                Convergence[FEcount-1] = Score
            if FEcount > MAXFEs:
                # print('itr', t)
                # print('FEc', FEcount)
                # print('lencon', len(Convergence))
                break
            if self.Sardine_pop > 1:
                Sailfish, Sardine, SailfishFitness, SardineFitness = self.Switch_Update(Sailfish, Sardine, SailfishFitness, SardineFitness)

            # Convergence[t + 1] = Score
            # if FEcount > MAXFEs:
            #     print('itr', t)
            #     print('FEc', FEcount)
            #     print('lencon', len(Convergence))
            #     break
        # np.savetxt('./MSFOResult/sailfishnum_F{}.csv'.format(self.fun_num), SailfishNumRecorder, delimiter=",")
        # plt.figure()
        # myfig = plt.gcf()
        # x = np.arange(0, self.Max_iteration, 1)
        # plt.plot(x, SailfishNumRecorder)
        # plt.xlabel("Iter")
        # plt.ylabel("sailfishnum")
        # myfig.savefig('./MSFOResult/V0/sailfishNum/SailfishnumF{}.png'.format(self.fun_num))
        # plt.close('all')
        return Score, BestFish, Convergence, con_conter

    def MSFO_Run(self, Recount):
        MAXFEs = 20000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        cons_total = np.zeros(7)
        best_solutions = np.zeros((Recount, self.dimensions))
        all_score = np.zeros(Recount)
        for i in range(Recount):
            s_score, s_bestfish, s_scoresum, con_c = self.MSFO_Searcher(i)
            score_sum += s_scoresum
            total_score[i] = s_scoresum
            cons_total += con_c
            best_solutions[i] = s_bestfish
            all_score[i] = s_score
        avg_scorecount = score_sum / Recount
        avg_con = cons_total / Recount
        # best_solutions = best_solutions / Recount
        np.savetxt('./WBDResult/con_result/sfo_con_counter_{}.csv'.format('WBD'), avg_con, delimiter=",")
        np.savetxt('./WBDResult/best_solution/sfo_best_solution_{}.csv'.format('WBD'), best_solutions, delimiter=",")
        np.savetxt('./WBDResult/all_score/SFO_all_score_{}.csv'.format('WBD'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig = plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./WBDResult/Pic/SFO_wbd_f{}.png'.format(self.fun_num))
        plt.close('all')
        # plt.figure()
        # myfig = plt.gcf()
        # x = np.arange(0, self.Max_iteration+1, 1)
        # plt.plot(x, avg_scorecount)
        # plt.xlabel("FEs")
        # plt.ylabel("Best Fitness")
        # myfig.savefig('./MSFOResult/V9EXP/Pic/PicF{}.png'.format(self.fun_num))
        # plt.close('all')

        return avg_scorecount, total_score


def main():
    recount = 30
    nu = 1
    msfo = MSFO(nu)
    avg_result, total_result = msfo.MSFO_Run(recount)
    np.savetxt('./WBDResult/avg_result/SFO_avg_{}.csv'.format('wbd'), avg_result, delimiter=",")
    np.savetxt('./WBDResult/total_result/SFO_total_{}.csv'.format('wbd'), total_result, delimiter=",")
    print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
