import random
import math
import numpy as np
import matplotlib.pyplot as plt

# EROAS V4
class ROA():
    def __init__(self, fun_num):
        self.Search_Agents = 50
        self.Max_iteration = 400
        self.dimensions = 11
        self.Uppernound = 100
        self.Lowerbound = -100
        tempdiv = (self.Uppernound-self.Lowerbound)/self.Search_Agents
        self.niche_r = tempdiv * 4.5
        self.fit_dis = 3
        self.fun_num = fun_num
        # self.EPSILON = 0.005             #depended on Max_iteration.plus is 0.5

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
        con_conter = np.zeros(10)

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
        cons_total = np.zeros(10)
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
        np.savetxt('./CSIDResult/con_result/EROAcon_counter_{}.csv'.format('CSID'), avg_con, delimiter=",")
        np.savetxt('./CSIDResult/best_solution/EROA_best_solution_{}.csv'.format('CSID'), best_solutions, delimiter=",")
        np.savetxt('./CSIDResult/all_score/EROA_all_score_{}.csv'.format('CSID'), all_score, delimiter=",")
        c = avg_scorecount
        plt.figure()
        myfig=plt.gcf()
        x = np.arange(0, MAXFEs, 1)
        plt.plot(x, c)
        plt.xlabel("# of fitness evaluations")
        plt.ylabel("best fitness")
        myfig.savefig('./CSIDResult/Pic/EROA_CSID_f{}.png'.format(self.fun_num))
        plt.close('all')
        return avg_scorecount, total_score


def main():
    recount = 30
    nu = 1
    ROAi = ROA(nu)
    c, b = ROAi.ROA_ReRun(recount)
    np.savetxt('./CSIDResult/avg_result/EROA_avg_{}.csv'.format('CSID'), c, delimiter=",")
    np.savetxt('./CSIDResult/total_result/EROA_total_{}.csv'.format('CSID'), b, delimiter=",")
    print("function:", nu, "is over.")
    return 0


if __name__=="__main__":
    main()
