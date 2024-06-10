import random
import math
import numpy as np
import matplotlib.pyplot as plt


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

    # def fitness_count(self, x):
    #     f = [0]
    #     cec17_test_func(x, f, self.dimensions, 1, self.fun_num)
    #     return f[0]

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
        penalty_function = [1000000, 1000000, 1000000, 1000000]
        penalty_term = 0
        for i in range(4):
            if cons_array[i] <= 0:
                con_conter[i] += 1
            else:
                penalty_term += penalty_function[i]
        fitness = self.PVD_obj(x) + penalty_term
        return fitness, con_conter

    def init_pos(self, search_agents, dimension, upperbound, lowerbound, Recount):
        Pos = np.zeros((search_agents, dimension))
        """
        :param X:
        0 <= X[0] <= 99,
        0 <= X[1] <= 99,
        10 <= X[2] <= 200,
        10 <= X[3] <= 200
        :return:
        """
        random.seed(2021 + self.fun_num + Recount)
        for i in range(search_agents):
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
        con_conter = np.zeros(4)

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
        cons_total = np.zeros(4)
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
        np.savetxt('./PVDResult/con_result/sfo_con_counter_{}.csv'.format('PVD'), avg_con, delimiter=",")
        np.savetxt('./PVDResult/best_solution/sfo_best_solution_{}.csv'.format('PVD'), best_solutions, delimiter=",")
        np.savetxt('./PVDResult/all_score/SFO_all_score_{}.csv'.format('PVD'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig = plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./PVDResult/Pic/SFO_PVD_f{}.png'.format(self.fun_num))
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
    np.savetxt('./PVDResult/avg_result/SFO_avg_{}.csv'.format('pvd'), avg_result, delimiter=",")
    np.savetxt('./PVDResult/total_result/SFO_total_{}.csv'.format('pvd'), total_result, delimiter=",")
    print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
