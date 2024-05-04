import os
import time
import math
import copy
import random

import numpy as np

class EA_Util:
    def __init__(self, name, pop_size, gen_size, eval_fun=None, max_gen=100, drop_set=[], sp_set=[], target='B'):

        self.name = name
        self.pop_size = pop_size
        self.gen_size = gen_size
        self.max_gen = max_gen
        self.drop_set = drop_set
        self.remain_set = []
        self.sp_set = sp_set
        self.target = target

        if eval_fun == None:
            raise Exception("Undefined Eval Function") 
        else:
            self.eval_fun = eval_fun
       
        for i in range(gen_size):
            if i not in self.drop_set:
                self.remain_set.append(i)

        self.cnt_remain = len(self.remain_set)
        print('Remain', self.cnt_remain)
        self.a = int(math.log(self.cnt_remain, 2)) // 2#log以2为底cnt_remain的对数
        self.b = int(math.log(self.cnt_remain, 2)) # 也可以用self.a + 1
        print('mutation interval: [%d, %d]' % (self.a, self.b)) # a和b表示的是变异的长度的变化范围

        self._init_pop()
    
    def _init_pop(self):
        population = []
        for _ in range(self.pop_size):
            individual = [1] * self.gen_size
            if self.target == 'B':# random init chromsome by 0 or 1
                for x in range(self.gen_size):
                    individual[x] = random.randint(0, 1)
            else:
                tmp = int(math.sqrt(self.gen_size))
                x = random.randint(tmp//2, tmp)
                individual = self._mutation(individual, cnt=x)
            for x in self.drop_set:#1是保留，0是drop
                individual[x] = 0
            for x in self.sp_set:
                individual[x] = 1
            population.append(individual)
        self.population = population
        self.fitness = [-1] * self.pop_size
    
    def _mutation(self, individual, cnt=1):
        new_chrom = individual.copy()
        for _ in range(cnt):
            s = random.randint(self.a, self.b)
            t = random.sample(self.remain_set, s)
            for x in t:
                new_chrom[x] = 1 - new_chrom[x]
        for x in self.sp_set:
            new_chrom[x] = 1
        return new_chrom
    
    def _eval_pop(self):#初始化fitness
        for x in range(self.pop_size):
            if self.fitness[x] < 0:
                acc = self.eval_fun(self.population[x])
                self.fitness[x] = acc
                # self.fitness[x] += 1.0 - (np.sum(self.population[x])*1.0/self.cnt_remain)
        
    def _reproduct(self, sur_cnt=None, mut_cnt=None):#重建种群 保留五分之一最好的，重新定义五分之二，根据bestavg定义五分之二
        temp_fit = self.fitness.copy()
        if sur_cnt == None:
            sur_cnt = self.pop_size // 5#保留个体数目
        if mut_cnt == None:#变异个体数目
            mut_cnt = sur_cnt * 2
        survival = []#survival和obsolete中存的都是individual的索引
        obsolete = []
        for _ in range(sur_cnt):#留下原来最好的5分之1
            cur_best = temp_fit.index(max(temp_fit))
            survival.append(cur_best)
            temp_fit[cur_best] = -1
        for x in range(self.pop_size):#剩下的全部丢弃
            if x not in survival:
                obsolete.append(x)
        print('    survival:', survival)
        print('    obsolete:', obsolete)
        bestavg = np.zeros(self.gen_size)
        for x in survival:
            bestavg += np.array(self.population[x], dtype=float)#最好的平均序列
        bestavg /= sur_cnt
        for i in range(mut_cnt):
            x = random.sample(survival, 1)[0]#survival是一个序列，所以取样之后也是得到一个序列，为了让x变成数，加一个[0]
            self.population[obsolete[i]] = self._mutation(self.population[x])
            self.fitness[obsolete[i]] = -1#变异得到的个体重新训练
        obs_len = len(obsolete)
        for i in range(mut_cnt, obs_len):
            individual = [1] * self.gen_size
            p = np.random.rand(self.gen_size)#随机生成gen_size个[0,1)之间的数
            for x in range(self.gen_size):
                individual[x] = 1 if p[x] <= bestavg[x] else 0
            for x in self.drop_set:
                individual[x] = 0
            for x in self.sp_set:
                individual[x] = 1
            self.population[obsolete[i]] = individual
            self.fitness[obsolete[i]] = -1
    
    def evolution(self):#得到迭代之后最好个体的索引
        best_fits = []
        self._eval_pop()
        print('init pop')
        best_fit = round(max(self.fitness), 4)
        tmp_fit = self.fitness.copy()
        best_fits.append(best_fit)
        for i in range(self.pop_size):
            tmp_fit[i] = round(tmp_fit[i], 4)
        print('  Best Fitness: %.4f' % (best_fit))
        print('  Pop Fitness:', tmp_fit)
        sur_cnt = 5#保留个体数目
        mut_cnt = 10#变异个体数目
        s_time = time.time()
        for gen in range(1, self.max_gen+1):
            print('%d evolution' % (gen))
            if gen == 10:
                mut_cnt = 20
            self._reproduct(sur_cnt=sur_cnt, mut_cnt=mut_cnt)
            self._eval_pop()
            best_fit = round(max(self.fitness), 4)
            tmp_fit = self.fitness.copy()
            best_fits.append(best_fit)
            for i in range(self.pop_size):
                tmp_fit[i] = round(tmp_fit[i], 4)
            print('  Best Fitness: %.4f' % (best_fit))
            print('  Pop Fitness:', tmp_fit)
            print('  time use:', time.time()-s_time)
            s_time = time.time()
        index = self.fitness.index(max(self.fitness))
        # with open("Checkpoint/model/Pruning_log/bestfitness.log", 'a') as f:
        #     f.writelines(["%s " % item for item in best_fits])
        #     f.write("\n")
        return index

  