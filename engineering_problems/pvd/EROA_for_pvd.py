import random
import math
import numpy as np
import matplotlib.pyplot as plt

# EROAS V4
class ROA():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 400
        self.dimensions = 4
        self.Uppernound = 100
        self.Lowerbound = -100
        tempdiv = (self.Uppernound-self.Lowerbound)/self.Search_Agents
        self.niche_r = tempdiv * 4.5
        self.fit_dis = 3
        self.fun_num = fun_num
        # self.EPSILON = 0.005             #depended on Max_iteration.plus is 0.5

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
        cur_nothost_arg_list = []                # 记录每个niche中非host点位置,长度不定，所以用list
        MAXFEs = 19999
        FEcount = 0
        Convergence = np.zeros(MAXFEs+1)
        con_conter = np.zeros(4)

        for r in range(0, self.Search_Agents):
            self.CheckIndi(Remora[r])
            fitness, con_conter = self.fitness_count(Remora[r], con_conter)
            FEcount += 1
            RemoraFitness[r] = fitness
            PbestFitness[r] = fitness
            # RemoraFitnessAllgen[0][r] = fitness
            if (fitness < Score):
                Score = fitness
                BestRemora = Remora[r].copy()
            Convergence[FEcount-1] = Score

        Remora, RemoraFitness = self.Sort_Remora(Remora, RemoraFitness)
        nicheremora = self.NicheMaker(Remora, RemoraFitness, self.niche_r, self.fit_dis)
        cur_host_arg = np.zeros(len(nicheremora), dtype=int)
        for hi in range(len(nicheremora)):
            tnicheremora = np.hstack(nicheremora[hi])        # 转成array
            one_cur_host_arg, nothost_arg = self.Find_host(tnicheremora, RemoraFitness)
            cur_host_arg[hi] = one_cur_host_arg
            cur_nothost_arg_list.append(nothost_arg)

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
                fitness, con_conter = self.fitness_count(Remora[c], con_conter)
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

                if fitness < Score:
                    Score = fitness
                    BestRemora = Remora[c].copy()
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

        return Score, BestRemora, Convergence, con_conter

    def ROA_ReRun(self, Recount):
        MAXFEs = 20000
        score_sum = np.zeros(MAXFEs)
        total_score = np.zeros((Recount, MAXFEs))
        cons_total = np.zeros(4)
        best_solutions = np.zeros((Recount, self.dimensions))
        all_score = np.zeros(Recount)
        for i in range(Recount):
            b_score, b_remora, score_count, con_c = self.ROA_searcher(i)
            score_sum += score_count
            total_score[i] = score_count
            cons_total += con_c
            best_solutions[i] = b_remora
            all_score[i] = b_score
        avg_scorecount = score_sum/Recount
        avg_con = cons_total / Recount
        # best_solutions = best_solutions / Recount
        np.savetxt('./PVDResult/con_result/EROAcon_counter_{}.csv'.format('PVD'), avg_con, delimiter=",")
        np.savetxt('./PVDResult/best_solution/EROA_best_solution_{}.csv'.format('PVD'), best_solutions, delimiter=",")
        np.savetxt('./PVDResult/all_score/EROA_all_score_{}.csv'.format('PVD'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./PVDResult/Pic/EROA_PVD_f{}.png'.format(self.fun_num))
        plt.close('all')
        return avg_scorecount, total_score

def main():
    recount = 30
    nu = 1
    ROAi = ROA(nu)
    c, b = ROAi.ROA_ReRun(recount)
    np.savetxt('./PVDResult/avg_result/EROA_avg_{}.csv'.format('PVD'), c, delimiter=",")
    np.savetxt('./PVDResult/total_result/EROA_total_{}.csv'.format('PVD'), b, delimiter=",")
    print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
