# from main import *
import numpy as np
import settings

s = settings.s
b = settings.b
loss = settings.loss
r = settings.r
w = settings.w

price_mat = settings.price_mat

kp_s = settings.kp_s
kn_s = settings.kn_s
kp_b = settings.kp_b
kn_b = settings.kn_b

zeta_p_s = settings.zeta_p_s
zeta_n_s = settings.zeta_n_s
zeta_p_b = settings.zeta_p_b
zeta_n_b = settings.zeta_n_b

target_price_sell = settings.target_price_sell
target_price_buy = settings.target_price_buy


def fyi(i,s,b, x, y, r, w, rho_0):
    yi = (sum(y*x*w) + (r-sum(x*w))*6)
    if yi > rho_0*r:
        vyi = kp_s[i]*(yi - rho_0*r)**zeta_p_s[i]
    else:
        vyi = -kn_s[i]*(rho_0*r - yi)**zeta_n_s[i]
    # print("vyi: ", vyi)
    return vyi


def fyj(j,s,b, x, y, w, rho_0):
    yj = sum(y*x)*w + (w-sum(x)*w)*12
    if yj < rho_0*w:
        vyj = kp_b[j]*(rho_0*w - yj)**zeta_p_b[j]
    else:
        vyj = -kn_b[j]*(yj - rho_0*w)**zeta_n_b[j]
    # print("vyj: ",vyj)
    return vyj


def obj_val_seller(v, price, day):
    x = v.reshape(-1)
    y = price.reshape(-1)  # v[(s*b):]

    fyii = sum(fyi(i, s=s, b=b, x=x[i*b:i*b+b], y=y[i*b:i*b+b], r=r[i,day], w=w[:,day],
                   rho_0=target_price_sell[i]) for i in range(s))

    return fyii

def obj_func(v, day):
    x = v[:(s*b)]
    # price_mat = price_matrix()

    y = price_mat.reshape(-1)  # v[(s*b):]
    # y = np.array([max(target_price_sell[i],target_price_buy[j]) for i in range(s) for j in range(b)])
    # y = np.array([target_price_sell[i] for i in range(s) for j in range(b)])

    # print(v)
    # print(x)
    # print(y)
    xj = x.reshape(s,b).T
    xj = xj.reshape(-1)

    yj = y.reshape(s,b).T
    yj = yj.reshape(-1)

    # fyii = sum(fyi(i, s = s, b= b, x=x[i*b:i*b+b], y = y[i*b:i*b+b], r=r[i,day], w = w[:,day],
    # rho_0 = target_price_sell[i]) for i in range(s))
    fyjj = sum(fyj(i, s = s, b= b, x=xj[i*s:i*s+s], y = yj[i*s:i*s+s], w = w[i,day], rho_0 = target_price_buy[i]) for i in range(b))

    return fyjj
    # return (fyii+fyjj) #-sum(y[i]*x[i]**2 for i in range(5))


def lossW(day):
    return (1+loss/100)*w[:,day]  # day=0


def constr(day):
    constraint_1 = np.zeros((s, s*b))
    LW = lossW(day)
    # constraint_1 = np.zeros((2*s,2*s*b))
    for i in range(s):
        constraint_1[i,i*b:i*b+b]= LW[i, :]
        # constraint_1 = constraint_1.reshape(-1)
    return constraint_1


def constr_2():
    constraint_2 = np.zeros((b, s*b))
    for i in range(b):
        for j in range(s*b):
            if j % b == i:
                constraint_2[i, j] = 1
            # constraint_1 = constraint_1.reshape(-1)
    return constraint_2


### self-made DEA

def checkConstraints(x, r, day):
    constr1 = constr(day)
    constr2 = constr_2()
    dim = len(x)
    x = np.where(loss.reshape(-1) > 2.5, 0, x)
    # x = np.where(price_mat.reshape(-1) >=12, 0, x)

    constraint_2_dot = np.dot(constr2, x)
    # x_mat = x[:int(dim/2)].reshape(s,b)
    x_mat = x.reshape(s, b)
    # print(x_mat)
    if not (constraint_2_dot < 1).all():
        # xT = x_mat.T
        for i in range(b):
            if sum(x_mat[:, i]) > 1:
                x_mat[:, i] = x_mat[:, i] / sum(x_mat[:, i])
        # x[:int(dim/2)] = x_mat.reshape(-1)
        x = x_mat.reshape(-1)

    # print(x)
    # print(constr1)
    constraint_1_dot = np.dot(constr1, x)
    # print(constraint_1_dot)
    for i in range(s):
        if constraint_1_dot[i] > r[i]:
            x_mat[i, :] = x_mat[i, :] * r[i] / constraint_1_dot[i]

    x = x_mat.reshape(-1)
    # x[:int(dim/2)] = x_mat.reshape(-1)
    # print('Hello!!!')
    return x


def checkBounds(x):
    dimensions = len(x)
    x = np.clip(x, 0, 1)
    # x[:int(dimensions/2)] = np.clip(x[:int(dimensions/2)], 0, 1)
    # x[int(dimensions/2):dimensions] = np.clip(x[int(dimensions/2):dimensions], 6, 12)
    return x


def de(fobj, bounds, r, day, mut=0.5, crossp=0.75, popsize=20, its=500):
    # print('Not supposed to happen!!')
    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    # pop_denorm[0] = [0.36, 0, 0, 0.36, 0, 0]
    for p in range(popsize):
        pop_denorm[p] = checkConstraints(pop_denorm[p], r, day)
        pop_denorm[p] = checkBounds(pop_denorm[p])

    fitness = np.asarray([fobj(ind, day) for ind in pop_denorm])
    best_idx = np.argmax(fitness)  # np.argmin(fitness)
    best = pop_denorm[best_idx]
    #print(pop_denorm)
    #print(best)

    for i in range(its):
        for j in range(popsize):
            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            # mutant = (a + mut * (b - c), 0, 1)#np.clip(a + mut * (b - c), 0, 1)
            mutant = (a + (mut + np.random.normal(0, 0.2)) * (b - c + np.random.normal(0, 0.2)))
            # mutant[:int(dimensions/2)] = np.clip(mutant[:int(dimensions/2)], 0, 1)
            # mutant[int(dimensions/2):dimensions] = np.clip(mutant[int(dimensions/2):dimensions], 6, 12)
            mutant = checkBounds(mutant)
            # print("Mutant after clipping:\n",mutant)
            # mutant[:dimension/2] = mutant[:dimensions/2] / np.linalg.norm(mutant[:dimensions/2])
            # mutant = checkConstraints(mutant, r)
            # mutant = checkBounds(mutant)
            # print("Mutant after constraint:\n",mutant)
            cross_points = np.random.rand(dimensions) < crossp
            # if all cross_points are false i.e all cross_points are >= crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = trial  # min_b + trial * diff
            trial_denorm = checkConstraints(trial_denorm, r,day)
            trial_denorm = checkBounds(trial_denorm)
            # trial_denorm[int(dimensions/2):] = np.where(trial_denorm[:int(dimensions/2)] == 0, 0, trial_denorm[int(dimensions/2):])

            f = fobj(trial_denorm, day)
            if f > fitness[j]:
                fitness[j] = f
                pop_denorm[j] = trial_denorm
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
            # if i in [0, 25, 50, 100, 200]:
            #     print('iteration-', i, 'pop-', j)
            #     # print(trial_denorm)
            #     print(pop_denorm[j])
        # if i in [0, 100, 250, 500, 999]:
        #     print(best_idx, best)
        yield best, fitness[best_idx]
