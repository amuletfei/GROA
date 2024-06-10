import random
import math
import numpy as np
# from cec2013.cec2013 import *
import matplotlib.pyplot as plt
from cec17_functions import cec17_test_func

# EROAS V3+elite
class ROA():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 100
        self.dimensions = 10
        self.Uppernound = 100
        self.Lowerbound = -100
        tempdiv = (self.Uppernound-self.Lowerbound)/self.Search_Agents
        self.niche_r = tempdiv * 4.5
        self.fit_dis = 3
        # self.Convergence = np.zeros(self.Max_iteration)
        self.fun_num = fun_num
        # self.EPSILON = 0.005             #depended on Max_iteration.plus is 0.5

    # def fitness_count(self, x):
    #     # f = [0]
    #     # cec17_test_func(x, f, self.dimensions, 1, self.fun_num)
    #     return x*x
    def fitness_count(self, x):
        f = [0]
        cec17_test_func(x, f, self.dimensions, 1, self.fun_num)
        return f[0]

    def init_pos(self, Recount):
        random.seed(2021 + self.fun_num + Recount)
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

    def NicheMaker(self, remora, remorafitness, disr, disf):
        nicheremora = []
        n = self.Search_Agents
        flagIn = np.zeros(n)
        for i in range(n):
            if flagIn[i] == 0:
                nichep = []
                nichep.append(i)
                flagIn[i] = 1
                for j in range(i + 1, n):
                    if flagIn[j] == 0:
                        dis = np.linalg.norm(remora[i]-remora[j])
                        # f_dis = abs(remorafitness[i] - remorafitness[j])
                        # print('dis,f_dis', dis, f_dis)
                        if dis < disr:
                            nichep.append(j)
                            flagIn[j] = 1
                    else:
                        continue
                if len(nichep) >= 1:
                    nicheremora.append(nichep)
            else:
                continue
        return nicheremora

    def Find_host(self, nicheramora, remorafitness):
        nichenum = len(nicheramora)
        # nothost_arg_list = []
        bestfit = float('inf')
        host_arg = 0
        hostarg_inniche = 0
        for i in range(nichenum):
            t = nicheramora[i]
            fitj = remorafitness[t]
            if fitj < bestfit:
                bestfit = fitj
                host_arg = int(t)
                hostarg_inniche = i
        nohostniche = np.delete(nicheramora, hostarg_inniche, axis=0)       # axis=0,删除行
        # nothost_arg = np.vstack(nothost_arg_list)
        # print('host_arg', host_arg)
        # print('nohostniche', nohostniche)
        return host_arg, nohostniche              # 返回一个host position和一个niche中所有非hots点的position

    def Reflect_Remora(self, One_Remora):
        # print('befor re', One_Remora)
        for i in range(self.dimensions):
            ref_cen = (self.Uppernound - self.Lowerbound)/self.Search_Agents
            One_Remora[i] = One_Remora[i] + ref_cen*random.randrange(1, 4)
        # print('after re', One_Remora)
        return One_Remora

    def Sort_Remora(self, Remora, RemoraFitness):
        recordRemora = np.zeros((self.Search_Agents, self.dimensions))
        recordFitness = np.zeros(self.Search_Agents)
        tempFitness = RemoraFitness
        for i in range(self.Search_Agents):
            argmax = np.argmax(tempFitness, axis=0)
            recordRemora[i] = Remora[argmax]
            recordFitness[i] = RemoraFitness[argmax]
            tempFitness[argmax] = float('-inf')

        return recordRemora, recordFitness

    def Crowded_Remora(self, Remora, RemoraFitness, iter):
        for i in range(1, self.Search_Agents-1):
            fitdis = abs(RemoraFitness[i-1] - RemoraFitness[i]) + abs(RemoraFitness[i+1] - RemoraFitness[i])
            dis = np.linalg.norm(Remora[i] - Remora[i - 1]) + np.linalg.norm(Remora[i] - Remora[i + 1])
            if fitdis < 2.56+0.02*(self.Max_iteration/iter) and dis <= self.niche_r:
                Remora[i] = self.Reflect_Remora(Remora[i])
        return Remora

    def ROA_searcher(self, Recount):
        Remora = self.init_pos(Recount)
        Prevgen = np.zeros((self.Max_iteration, self.Search_Agents, self.dimensions))        # 储存每代位置
        Prevgen[0] = Remora.copy()
        Score = float('inf')
        BestRemora = np.zeros((1, self.dimensions))
        RemoraFitness = np.zeros(self.Search_Agents)  # 记录当前iteration中所有search_agents的fitness
        PbestFitness = np.zeros(self.Search_Agents)   # 记录所有个体，过去最好的fitness
        # BestRecorder = []              # 记录每一代中所有fitness最好个体
        # PbestPos = Remora                 # 记录个体历史最好位置
        # pre_host_arg = []                 # 记录过去host的位置
        # cur_host_arg = []                 # 当前host位置
        cur_nothost_arg_list = []                # 记录每个niche中非host点位置,长度不定，所以用list
        # RemoraFitnessAllgen = np.zeros((self.Max_iteration+1, self.Search_Agents))
        # host_recorder_list = []  # 记录所有世代的host下标
        MAXFEs = 4999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(Remora[r])
            fitness = self.fitness_count(Remora[r])
            FEcount += 1
            RemoraFitness[r] = fitness
            PbestFitness[r] = fitness
            # RemoraFitnessAllgen[0][r] = fitness
            if (fitness < Score):
                Score = fitness
                BestRemora[0] = Remora[r].copy()
            Convergence[FEcount-1] = Score


        Remora, RemoraFitness = self.Sort_Remora(Remora, RemoraFitness)
        nicheremora = self.NicheMaker(Remora, RemoraFitness, self.niche_r, self.fit_dis)
        cur_host_arg = np.zeros(len(nicheremora), dtype=int)
        # pre_host_arg = np.zeros(len(nicheremora), dtype=int)
        for hi in range(len(nicheremora)):
            tnicheremora = np.hstack(nicheremora[hi])        # 转成array
            one_cur_host_arg, nothost_arg = self.Find_host(tnicheremora, RemoraFitness)
            cur_host_arg[hi] = one_cur_host_arg
            # pre_host_arg[hi] = one_cur_host_arg
            cur_nothost_arg_list.append(nothost_arg)
        # host_recorder_list.append(cur_host_arg)
        # host_recorder0 = np.hstack(host_recorder_list)
        # addr1 = './resultNROAV5/txthost/host_recorder{}/'.format(self.fun_num)
        # addr1 = addr1 + 'host_gen{}.csv'.format(0)
        # np.savetxt(addr1, host_recorder0, delimiter=",")

        # print('Gen0,curhost', cur_host_arg)
        # print('Gen0,curnohost', cur_nothost_arg_list)


        for i in range(1, self.Max_iteration):
            # print('In iter,cur_host len', len(cur_host_arg))
            # host_recorder_list = []
            for j in range(len(cur_host_arg)):
                rpos = cur_host_arg[j]
                if RemoraFitness[rpos] == PbestFitness[rpos]:
                    for d in range(self.dimensions):
                        Remora[rpos][d] = Remora[rpos][d] + random.gauss(0, 0.75)   # v2
                        # Remora[j][d] = Remora[j][d] + random.gauss(0, 1.5)

                cn = random.randint(0, 2)
                if cn == 0:
                    r_best = np.argmin(RemoraFitness, axis=0)
                    current_best = Remora[r_best].copy()
                    a = -(1 + i / self.Max_iteration)  # equation 7
                    alpha = random.random() * (a - 1) + 1  # equation 6
                    d = abs(current_best - Remora[rpos, :])  # equation 8
                    Remora[rpos] = d * np.exp(alpha) * math.cos(2 * math.pi * alpha) + Remora[rpos]  # equation 5
                if cn == 1:
                    # a = 2 - (2 * i / self.Max_iteration)
                    # C = 2 * random.random()
                    # A = 2 * a * random.random() - a
                    # D = abs(C * BestRemora - Remora[rpos])
                    # Remora[rpos] = BestRemora - A * D
                    r_best = np.argmin(RemoraFitness, axis=0)
                    current_best = Remora[r_best].copy()
                    v = 2*(1-i/self.Max_iteration)
                    b = 2*v*random.random()-v
                    c = 0.1
                    a = b*(Remora[rpos]-c*current_best)
                    Remora[rpos] = Remora[rpos]+a
                if cn == 2:
                    rm = random.randint(0, self.Search_Agents-1)
                    Remora[rpos] = BestRemora - (random.random() * ((BestRemora + Remora[rm]) / 2) - Remora[rm])

            for k in range(len(cur_nothost_arg_list)):
                # print('curnohostlist', cur_nothost_arg_list)
                # print('curK', k)
                # print('lenlist', len(cur_nothost_arg_list[0]))
                if len(cur_nothost_arg_list[k]) > 0:
                    one_nothost_arg = np.hstack(cur_nothost_arg_list[k])
                    cur_host = Remora[cur_host_arg[k]]
                    if len(one_nothost_arg) > 3:
                        select_num = len(one_nothost_arg) - 3
                        select_arg = []
                        for s in range(select_num):
                            select_arg.append(random.randint(0, len(one_nothost_arg)))
                        for m in range(len(one_nothost_arg)):
                            nhpos = one_nothost_arg[m]
                            if m in select_arg:
                                Remora[nhpos] = self.Reflect_Remora(Remora[nhpos])
                            else:
                                if random.randint(0, 1) == 0:  # v3
                                    Remora[nhpos] = cur_host + 1.25 * math.cos(math.pi * random.random())
                                else:
                                    if i <= 1:
                                        PreR = Prevgen[0]
                                    else:
                                        PreR = Prevgen[i - 1]
                                    rn = random.uniform(0, 1) * 2 - 1
                                    Remora[nhpos] = cur_host + abs(cur_host - PreR[cur_host_arg[k]]) * math.cos(
                                        math.pi * rn * 2)
                    else:
                        for n in range(len(one_nothost_arg)):
                            nhpos = one_nothost_arg[n]
                            if random.randint(0, 1) == 0:  # v3
                                Remora[nhpos] = cur_host + 1.25 * math.cos(math.pi * random.random())
                            else:
                                if i <= 1:
                                    PreR = Prevgen[0]
                                else:
                                    PreR = Prevgen[i - 1]
                                rn = random.uniform(0, 1) * 2 - 1
                                Remora[nhpos] = cur_host + abs(cur_host - PreR[cur_host_arg[k]]) * math.cos(
                                    math.pi * rn * 2)

            for c in range(0, self.Search_Agents):
                if FEcount > MAXFEs:
                    break
                self.CheckIndi(Remora[c])
                fitness = self.fitness_count(Remora[c])
                temp_fitness = fitness
                if temp_fitness < RemoraFitness[c]:
                    RemoraFitness[c] = fitness
                else:
                    fitness = RemoraFitness[c]
                    Remora[c] = Prevgen[i-1][c].copy()
                # RemoraFitnessAllgen[i][c] = fitness
                FEcount += 1
                # print('fitness', fitness)
                # print('Score', Score)
                if fitness < PbestFitness[c]:
                    PbestFitness[c] = fitness

                if (fitness < Score):
                    Score = fitness
                    BestRemora[0] = Remora[c].copy()
                Convergence[FEcount-1] = Score

            Remora, RemoraFitness = self.Sort_Remora(Remora, RemoraFitness)
            nicheremora = self.NicheMaker(Remora, RemoraFitness, self.niche_r, self.fit_dis)
            # pre_host_arg = cur_host_arg
            cur_host_arg = np.zeros(len(nicheremora), dtype=int)
            cur_nothost_arg_list = []
            for hi in range(len(nicheremora)):
                tnicheremora = np.hstack(nicheremora[hi])        # 转成array
                one_cur_host_arg, nothost_arg = self.Find_host(tnicheremora, RemoraFitness)
                cur_host_arg[hi] = one_cur_host_arg
                cur_nothost_arg_list.append(nothost_arg)
            Prevgen[i] = Remora.copy()
            if FEcount > MAXFEs:
                break
            # self.Convergence[i] = Score

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
        # d = total_score
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./EROASResult/V4/10D/Pic/EROASV4_f{}.png'.format(self.fun_num))
        # # plt.show()
        # np.savetxt('./resultNROAV4/txtavg/avg_score{}.csv'.format(self.fun_num), avg_scorecount, delimiter=",")
        # np.savetxt('./resultNROAV4/txttotal/total_score{}.csv'.format(self.fun_num), d, delimiter=",")
        plt.close('all')
        # return avg_scorecount, total_score
        # opc = 0
        # real_gopc = 0
        # NSR = 0
        # for i in range(Recount):
        #     t_opc, t_real_gopc = self.ROA_searcher()
        #     opc += t_opc
        #     real_gopc += t_real_gopc
        #     if t_opc == t_real_gopc:
        #         NSR = NSR + 1
        #
        # PR = opc / real_gopc
        # SR = NSR / Recount
        return avg_scorecount, total_score

def main():
    # repeating times:recount
    recount = 30
    for nu in range(1, 31):
        if nu != 2:
            ROAi = ROA(nu)
            c, b = ROAi.ROA_ReRun(recount)
            # print(f"In the function {nu},PR is {pr},SR is {sr}")
            # print(f"function:", nu, "is over.")
            np.savetxt('./EROASResult/V4/10D/avgResult/avg_score{}.csv'.format(nu), c, delimiter=",")
            np.savetxt('./EROASResult/V4/10D/totalResult/totalResult{}.csv'.format(nu), b, delimiter=",")
            print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
