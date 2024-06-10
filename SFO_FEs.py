import random
import math
import numpy as np
import matplotlib.pyplot as plt
from cec17_functions import cec17_test_func

# with FEs
class SFO():
    def __init__(self,  fun_num):
        self.Sailfish_pop = 50
        self.Sardine_pop = 50
        self.Max_iteration = 500        # 10D：100，30D：300，50D：500
        self.Uppernound = 100
        self.Lowerbound = -100
        self.dimensions = 50
        self.EPSILON = 0.001         # 10D：0.005，30D：0.0016，50D：0.001
        self.fun_num = fun_num
        # self.Convergence = np.zeros(self.Max_iteration+1)

    def fitness_count(self, x):
        f = [0]
        cec17_test_func(x, f, self.dimensions, 1, self.fun_num)
        return f[0]

    def init_pos(self, Search_Agents, Recount):
        random.seed(2021 + self.fun_num + Recount)
        Pos = np.zeros((Search_Agents, self.dimensions))
        for i in range(Search_Agents):
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

    def SFO_Searcher(self, Recount):
        Sailfish = self.init_pos(self.Sailfish_pop, Recount)
        Sardine = self.init_pos(self.Sardine_pop, Recount)
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
        MAXFEs = 24999
        # self.Convergence = np.zeros(self.Max_iteration+1)
        Convergence = np.zeros(MAXFEs+1)
        FEcount = 0

        for i in range(self.Sailfish_pop):
            self.CheckIndi(Sailfish[i])
            fitness = self.fitness_count(Sailfish[i])
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
            fitness = self.fitness_count(Sardine[i])
            # FEcount += 1
            SardineFitness[i] = fitness
            # if fitness < Score:
            #     Score = fitness
            #     BestFish = Sardine[i].copy()
            if fitness < SardineScore:
                SardineScore = fitness
                BestSardine = Sardine[i].copy()
            # Convergence[FEcount-1] = Score

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
                fitness = self.fitness_count(Sailfish[j])
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
                fitness = self.fitness_count(Sardine[k])
                # FEcount += 1
                if fitness < SardineScore:
                    SardineScore = fitness
                    BestSardine = Sardine[k].copy()
                # if fitness < Score:
                #     Score = fitness
                #     BestFish = Sardine[k].copy()
                PreSardineFitness[k] = SardineFitness[k]
                SardineFitness[k] = fitness
                # Convergence[FEcount-1] = Score
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
        return Score, BestFish, Convergence

    def SFO_Run(self, Recount):
        MAXFEs = 25000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        for i in range(Recount):
            self.Sardine_pop = 50
            s_score, s_bestfish, s_scoresum = self.SFO_Searcher(i)
            score_sum += s_scoresum
            total_score[i] = s_scoresum
        avg_scorecount = score_sum/Recount
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./SFOResult/50D/Pic/SFOFEs_f{}.png'.format(self.fun_num))
        # plt.show()
        plt.close('all')

        return avg_scorecount, total_score


def main():
    recount = 30
    for nu in range(1, 31):
        if nu != 2:
            f_num = nu
            msfo = SFO(nu)
            avg_result, total_result = msfo.SFO_Run(recount)
            np.savetxt('./SFOResult/50D/avgResult/avg_score{}.csv'.format(f_num), avg_result, delimiter=",")
            np.savetxt('./SFOResult/50D/totalResult/totalResult{}.csv'.format(f_num), total_result, delimiter=",")
            print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
