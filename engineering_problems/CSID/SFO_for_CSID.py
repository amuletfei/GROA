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
        self.dimensions = 11
        self.EPSILON = 0.001
        self.fun_num = fun_num
        # self.Convergence = np.zeros(self.Max_iteration+1)

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

    def init_pos(self, search_agents, dimension, upperbound, lowerbound, Recount):
        Pos = np.zeros((search_agents, dimension))
        """
        :param x:
        0.5 <= X[0],x[1],x[2],x[3],x[4],x[5],x[6] <= 1.5,
        0.192 <= X[7],x[8] <= 0.345,
        -30 <= X[9],x[10] <= 30,
        """
        random.seed(2021 + self.fun_num + Recount)
        for i in range(search_agents):
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
        con_conter = np.zeros(10)

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

        return Score, BestFish, Convergence, con_conter

    def MSFO_Run(self, Recount):
        MAXFEs = 20000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        cons_total = np.zeros(10)
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
        np.savetxt('./CSIDResult/con_result/sfo_con_counter_{}.csv'.format('CSID'), avg_con, delimiter=",")
        np.savetxt('./CSIDResult/best_solution/sfo_best_solution_{}.csv'.format('CSID'), best_solutions, delimiter=",")
        np.savetxt('./CSIDResult/all_score/SFO_all_score_{}.csv'.format('CSID'), all_score, delimiter=",")

        c = avg_scorecount
        plt.figure()
        myfig = plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./CSIDResult/Pic/SFO_CSID_f{}.png'.format(self.fun_num))
        plt.close('all')

        return avg_scorecount, total_score


def main():
    recount = 30
    nu = 1
    msfo = MSFO(nu)
    avg_result, total_result = msfo.MSFO_Run(recount)
    np.savetxt('./CSIDResult/avg_result/SFO_avg_{}.csv'.format('CSID'), avg_result, delimiter=",")
    np.savetxt('./CSIDResult/total_result/SFO_total_{}.csv'.format('CSID'), total_result, delimiter=",")
    print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
