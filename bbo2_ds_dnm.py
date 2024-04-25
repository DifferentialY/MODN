import copy
import torch
import torch.nn as nn
import numpy as np
import random
import scipy.linalg
import multiprocessing as mp
from multiprocessing import Pool
from RouletteWheelSelection import RouletteWheelSelection


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def md_DNM(train_data, net, w, q, ws, qs, class_num, ds):
    setup_seed(1)
    M = net.M
    k = net.k
    ks = net.ks
    somafunc = net.somafunc

    P = torch.transpose(train_data, 0, 1)
    train_num = np.size(np.array(train_data), 0)

    V = torch.zeros(1, class_num)
    O = torch.zeros(class_num)
    Q = torch.zeros(train_num, class_num)
    for h in range(train_num):
        insi = torch.transpose(P[:, h].repeat(M, 1), 0, 1)
        Y = 1. / (1 + torch.exp(-k * (w * insi - q)))
        Z = torch.prod(Y, 0)
        for i in range(class_num):
            V[0, i] = torch.sum(Z * ds[i, :])

            # sigmoid
            if somafunc == 'sigmoid':
                O[i] = 1. / (1 + torch.exp(-ks * (ws[i] * V[0, i] - qs[i])))
            elif somafunc == 'tanh':
                O[i] = torch.tanh(ks * (ws[i] * V[0, i] - qs[i]))
            elif somafunc == 'relu':
                O[i] = torch.relu(ks * (ws[i] * V[0, i] - qs[i]))
            elif somafunc == 'leakyrelu':
                O[i] = ks * (ws[i] * V[0, i] - qs[i])
                if O[i] <= 0:
                    O[i] = 0.1 * O[i]

        O = torch.exp(O) / torch.sum(torch.exp(O))  # softmax
        Q[h, :] = O

    return Q


def BBO_pop(pop, popindex, M, D, class_num, train_dim, train_data, net, train_target, train_num):
    w = pop[popindex, 0:train_dim * M]
    q = pop[popindex, train_dim * M:D]
    ws = pop[popindex, D:D + class_num]
    qs = pop[popindex, D + class_num:D + 2 * class_num]
    w = torch.reshape(w, (train_dim, M))
    q = torch.reshape(q, (train_dim, M))
    ds = pop[popindex, D + 2 * class_num:D + 2 * class_num + class_num * M]
    ds = torch.reshape(ds, (class_num, M))
    ds = (torch.sign(ds) + 1) / 2  # binaryzation
    train_fit = md_DNM(train_data, net, w, q, ws, qs, class_num, ds)

    # cross-entropy
    er = - train_target * torch.log(train_fit)
    fitness = torch.sum(er) / train_num
    fitness = torch.FloatTensor(fitness)
    return fitness


def bbo2_ds_dnm(net, train_data, train_target, class_num, random_seed):
    setup_seed(random_seed)
    # global mu
    popsize = net.popsize
    D = net.D
    Maxit = net.iter
    Maxit_d = 10
    M = net.M
    dtype = net.dtype
    fitness = torch.zeros(1, popsize)
    train_num = np.size(np.array(train_data), 0)
    train_dim = np.size(np.array(train_data), 1)

    # initial population
    lu = np.vstack((-1 * torch.ones(1, D), torch.ones(1, D)))
    lu = torch.FloatTensor(lu)

    # BBO train
    KeepRate = torch.Tensor([0.2])
    VarSize = [1, D]
    VarMin = lu[0, 0]
    VarMax = lu[1, 0]
    nKeep = torch.round(KeepRate * popsize)
    nKeep = nKeep.int()
    nNew = popsize - nKeep

    # Migration Rates
    mu = torch.linspace(1, 0, popsize)
    mu = mu.reshape(1, popsize)
    lamd = 1 - mu

    alpha = 0.9

    pMutation = 0.1

    sigma = 0.02 * (lu[1, 0] - lu[0, 0])

    pos = -1 + 2 * torch.rand(popsize, D)
    somapara = torch.rand(popsize, 2 * class_num)
    position = torch.cat((pos, somapara), 1)

    if dtype == 'same':
        # enforce all selection vectors to be same input
        ds = torch.ones(1, class_num * M)
        ds = ds.repeat(popsize, 1)
    elif dtype == 'different':
        # enforce all selection vectors to be different input
        d = torch.ones(int(M / class_num))
        ds = torch.ones(int(M / class_num))
        for _ in range(class_num - 1):
            ds = scipy.linalg.block_diag(ds, d)
        ds = torch.reshape(torch.FloatTensor(ds), (1, class_num * M))
        ds = ds.repeat(popsize, 1)
    elif dtype == 'random':
        ds = -1 + 2 * torch.rand(popsize, class_num * M)

    position = torch.cat((position, ds), 1)
    pop = copy.deepcopy(position)

    for popindex in range(popsize):
        w = pop[popindex, 0:train_dim * M]
        q = pop[popindex, train_dim * M:D]
        ws = pop[popindex, D:D + class_num]
        qs = pop[popindex, D + class_num:D + 2 * class_num]
        w = torch.reshape(w, (train_dim, M))
        q = torch.reshape(q, (train_dim, M))
        ds = pop[popindex, D + 2 * class_num:D + 2 * class_num + class_num * M]
        ds = torch.reshape(ds, (class_num, M))
        ds = (torch.sign(ds) + 1)/2  # binaryzation
        train_fit = md_DNM(train_data, net, w, q, ws, qs, class_num, ds)

        # cross-entropy
        er = - train_target * torch.log(train_fit)
        fitness[0, popindex] = torch.sum(er)/train_num
        fitness = torch.FloatTensor(fitness)

    cost = copy.deepcopy(fitness)

    # Sort new population
    cost, cost_index = torch.sort(cost)
    for ind in range(popsize):
        position[ind, :] = pop[cost_index[0, ind], :]

    dspop = position[:, D + 2 * class_num:D + 2 * class_num + class_num * M]
    bestcost = torch.zeros(2, Maxit)

    for it in range(Maxit_d):
        newdspop = copy.deepcopy(dspop)
        newcost = copy.deepcopy(cost)
        for i in range(popsize):
            for k in range(class_num * M):
                # Migration
                if random.random() < lamd[0, i]:
                    # Emmigration probabilities

                    EP = copy.deepcopy(mu)
                    EP[0, i] = 0
                    EP = EP / torch.sum(EP)

                    # Select source habitat
                    j = RouletteWheelSelection(EP)

                    # Migration
                    # newdspop[i, k] = dspop[i, k] + alpha * (dspop[j, k] - dspop[i, k])
                    newdspop[i, k] = dspop[j, k]

                if torch.rand(1) <= pMutation:
                    newdspop[i, k] = 2 * torch.rand(1) - 1
                    # newpop[i, k] = lu[0, 0] + (lu[1, 0]-lu[0, 0]) * random.random()

            # Apply lower and upper bound limits
            # newpop[i, :] = torch.where(newpop[i, :] < VarMin, VarMin, newpop[i, :])
            # newpop[i, :] = torch.where(newpop[i, :] > VarMax, VarMax, newpop[i, :])
            # newpop[i, :] = np.clip(newpop[i, :], VarMin, VarMax)
            # newpop[i, :D + 2 * class_num] = np.clip(newpop[i, :D + 2 * class_num], VarMin, VarMax)
            newdspop[i, :] = np.clip(newdspop[i, :], VarMin, VarMax)

        for popindex in range(popsize):
            w = position[0, 0:train_dim * M]
            q = position[0, train_dim * M:D]
            ws = position[0, D:D + class_num]
            qs = position[0, D + class_num:D + 2 * class_num]
            w = torch.reshape(w, (train_dim, M))
            q = torch.reshape(q, (train_dim, M))
            ds = newdspop[popindex, :]
            ds = torch.reshape(ds, (class_num, M))
            ds = (torch.sign(ds) + 1) / 2  # binaryzation
            train_fit = md_DNM(train_data, net, w, q, ws, qs, class_num, ds)

            # cross-entropy
            er = - train_target * torch.log(train_fit)
            fitness[0, popindex] = torch.sum(er)/train_num
            fitness = torch.FloatTensor(fitness)

        newcost = copy.deepcopy(fitness)

        newcost, newcost_index = torch.sort(newcost)
        copdspop = copy.deepcopy(newdspop)
        for ind in range(popsize):
            newdspop[ind, :] = copdspop[newcost_index[0, ind], :]

        # Select next iteration population
        dspop = torch.cat((dspop[0:nKeep, :], newdspop[0:nNew, :]), dim=0)
        cost = torch.cat((cost[:, 0:nKeep], newcost[:, 0:nNew]), dim=1)

        cost, cost_index = torch.sort(cost)
        copdsppop = copy.deepcopy(dspop)
        for ind in range(popsize):
            dspop[ind, :] = copdsppop[cost_index[0, ind], :]

        bestcost[0, it] = cost[0, 0]
        print('DSit=', it, 'LOSS=', bestcost[0, it])

        ds = dspop[0, :]
        pop = torch.cat((position[:, 0:D + 2 * class_num], ds.repeat(popsize, 1)), dim=1)
        for popindex in range(popsize):
            w = pop[popindex, 0:train_dim * M]
            q = pop[popindex, train_dim * M:D]
            ws = pop[popindex, D:D + class_num]
            qs = pop[popindex, D + class_num:D + 2 * class_num]
            w = torch.reshape(w, (train_dim, M))
            q = torch.reshape(q, (train_dim, M))
            ds = pop[popindex, D + 2 * class_num:D + 2 * class_num + class_num * M]
            ds = torch.reshape(ds, (class_num, M))
            ds = (torch.sign(ds) + 1) / 2  # binaryzation
            train_fit = md_DNM(train_data, net, w, q, ws, qs, class_num, ds)

            # cross-entropy
            er = - train_target * torch.log(train_fit)
            fitness[0, popindex] = torch.sum(er) / train_num
            fitness = torch.FloatTensor(fitness)

        cost = copy.deepcopy(fitness)

        # Sort new population
        cost, cost_index = torch.sort(cost)
        for ind in range(popsize):
            position[ind, :] = pop[cost_index[0, ind], :]

    pop = copy.deepcopy(position)
    for it in range(Maxit):
        newpop = copy.deepcopy(pop)
        newcost = copy.deepcopy(cost)
        for i in range(popsize):
            for k in range(D + 2 * class_num):
                # Migration
                if random.random() < lamd[0, i]:
                    # Emmigration probabilities

                    EP = copy.deepcopy(mu)
                    EP[0, i] = 0
                    EP = EP / torch.sum(EP)

                    # Select source habitat
                    j = RouletteWheelSelection(EP)

                    # Migration
                    newpop[i, k] = pop[i, k] + alpha * (pop[j, k] - pop[i, k])
                    # newpop[i, k] = pop[j, k]

                if torch.rand(1) <= pMutation:
                    newpop[i, k] = newpop[i, k] + sigma * torch.randn(1)
                    # newpop[i, k] = lu[0, 0] + (lu[1, 0]-lu[0, 0]) * random.random()

        for popindex in range(popsize):
            w = newpop[popindex, 0:train_dim * M]
            q = newpop[popindex, train_dim * M:D]
            ws = newpop[popindex, D:D + class_num]
            qs = newpop[popindex, D + class_num:D + 2 * class_num]
            w = torch.reshape(w, (train_dim, M))
            q = torch.reshape(q, (train_dim, M))
            ds = newpop[popindex, D + 2 * class_num:D + 2 * class_num + class_num * M]
            ds = torch.reshape(ds, (class_num, M))
            ds = (torch.sign(ds) + 1) / 2  # binaryzation
            train_fit = md_DNM(train_data, net, w, q, ws, qs, class_num, ds)

            # cross-entropy
            er = - train_target * torch.log(train_fit)
            fitness[0, popindex] = torch.sum(er) / train_num
            fitness = torch.FloatTensor(fitness)

        newcost = copy.deepcopy(fitness)

        # Sort new population
        newcost, newcost_index = torch.sort(newcost)
        coppop = copy.deepcopy(newpop)
        for ind in range(popsize):
            newpop[ind, :] = coppop[newcost_index[0, ind], :]

        # Select next iteration population
        pop = torch.cat((pop[0:nKeep, :], newpop[0:nNew, :]), dim=0)
        cost = torch.cat((cost[:, 0:nKeep], newcost[:, 0:nNew]), dim=1)

        # Sort population
        cost, cost_index = torch.sort(cost)
        copppop = copy.deepcopy(pop)
        for ind in range(popsize):
            pop[ind, :] = copppop[cost_index[0, ind], :]

        bestcost[1, it] = cost[0, 0]
        print('it=', it, 'LOSS=', bestcost[1, it])

    # Return
    best = cost[0, 0]
    index = 0
    best_population = newpop[index, :]
    best_population = torch.reshape(best_population, (1, D + 2 * class_num + class_num * M))
    w = best_population[0, 0:train_dim * M]
    q = best_population[0, train_dim * M:D]
    ws = best_population[0, D:D + class_num]
    qs = best_population[0, D + class_num:D + 2 * class_num]
    w = torch.reshape(w, (train_dim, M))
    q = torch.reshape(q, (train_dim, M))
    ds = best_population[0, D + 2 * class_num:D + 2 * class_num + class_num * M]
    ds = torch.reshape(ds, (class_num, M))
    # ds = (torch.sign(ds) + 1) / 2

    class outF:
        def __init__(self, convergence, w, q):
            self.convergence = bestcost
            self.w = w
            self.q = q

    out = outF(bestcost, w, q)
    best = copy.deepcopy(bestcost)
    return w, q, ws, qs, ds, best, out
