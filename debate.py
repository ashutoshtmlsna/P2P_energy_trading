import numpy as np
import pandas as pd
import random
import de_optimizer

import matplotlib.pyplot as plt
import time
import resource
import os

import settings
import p2p_pqr_dqn

s = settings.s
b = settings.b
loss = settings.loss
r = settings.r
w = settings.w

price_mat = settings.price_mat
bounds = settings.bounds

target_price_sell = settings.target_price_sell
target_price_buy = settings.target_price_buy
action_size = settings.action_size
state_size = settings.state_size

eps_decay = settings.eps_decay
gamma = settings.gamma
lr = settings.lr


def debate(pricing="pqr"):
    obj_vals = pd.DataFrame(columns=['buyer', 'seller', 'price', 'amt'])
    obj_buyer = []
    obj_seller = []
    # avg_price_rule = pd.DataFrame(columns=['price', 'amt'])
    avg_price = []
    avg_amt = []
    selling_price = np.array(target_price_sell)
    sell_p = []
    # sell_r = []
    for i in range(s):
        # sell_r.append([])
        sell_p.append([])
    psize = 20
    eps = 1

    ind_prices = pd.DataFrame(columns=['seller' + str(i) for i in range(s)])

    time_duration = 365

    if pricing == "dqn":
        print("running ProDQN")
        agent = [p2p_pqr_dqn.Agent(state_size=1, action_size=action_size, seed=None, h1=64, h2=32,i=i) for i in range(s)]
    else:
        ## Q-learning ##
        print("running PQR")
        Q = np.zeros((s, state_size, action_size))

    for day in range(time_duration):
        time_start = time.perf_counter()
        obj_func = de_optimizer.obj_func
        result = de_optimizer.de(obj_func, bounds, r=r[:, day], day=day, popsize=psize, its=20000)

        solution = list(result)[-1]
        obj_val = solution[-1]
        solution = solution[0]
        # print(day, solution, obj_val)

        obj_buyer.append(obj_val)

        sol_x = solution[:s * b]
        sol_x = sol_x.reshape(s, b)
        # sol_y = sol_y.reshape(s,b)
        # print("Fraction of amt of energy:\n", sol_x)

        time_elapsed = (time.perf_counter() - time_start)
        memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
        print("%5.1f secs %5.1f MByte" % (time_elapsed, memMb))

        sum_i_xij = np.sum(sol_x, axis=0)
        buyers_demand_fulfilled = np.sum(sol_x * w[:, day], axis=0)
        unsatisfied_demand = w[:, day] - buyers_demand_fulfilled
        # print(sum_i_xij)
        # print('Fulfilled Demand:\n', buyers_demand_fulfilled)
        # print('Unsatisfied Demand:\n', unsatisfied_demand)
        sellers_cap_sold = np.dot(sol_x, w[:, day])
        # print(sellers_cap_sold)
        avg_price.append(sum(selling_price * sellers_cap_sold) / sum(sellers_cap_sold))
        avg_amt.append(sum(sellers_cap_sold))

        cum_reward_daily = sum(selling_price * sellers_cap_sold) / 1000
        obj_seller.append(cum_reward_daily)

        print(day, obj_val, obj_seller)

        # selling_price = learn_price(selling_price, sellers_cap_sold, day)
        for i in range(s):
            if pricing == "dqn":
                selling_price[i] = p2p_pqr_dqn.dqn(i, agent[i], selling_price[i], sellers_cap_sold[i], day=day,eps=eps)
            else:
                selling_price[i] = p2p_pqr_dqn.q_learn_price(i, Q, selling_price[i], sellers_cap_sold[i], day=day,eps=eps)
            sell_p[i] += [selling_price[i]]
        price_mat = np.array([selling_price[i] for i in range(s) for j in range(b)]).reshape(s, b)
        # 0.965 best so far
        eps = max(eps_decay * eps, 0.05)  # if eps > 0.05 else 0.05

        ind_prices.loc[day, :] = selling_price

    # timestr = time.strftime("%y_%m_%d-%H_%M_%S")
    # timestr = time.strftime("%y_%m_%d")
    # expt_dir = f"de_expt/gen/expt_{timestr}"
    expt_dir = f"de_expt/expt16_22_06_09"
    if not os.path.exists(expt_dir):
        os.mkdir(expt_dir)
    # plt.savefig(expt_dir+"/avg_ind_price_obj_plot_dqnk" + str(s) + "_" + str(b) + "_" + timestr + ".png")

    obj_vals['buyer'] = obj_buyer
    obj_vals['seller'] = obj_seller  # obj_vals_seller
    # obj_vals['reward'] = cum_reward
    obj_vals['price'] = avg_price
    for i in range(s):
        ind_prices['seller' + str(i)] = sell_p[i]
    if pricing == "dqn":
        obj_vals.to_csv(expt_dir+"/obj_vals_dqn_"+str(s)+"_"+str(b)+".csv", sep=",", float_format='%.6f')
        ind_prices.to_csv(expt_dir+"/ind_prices_dqn_"+str(s)+"_"+str(b)+".csv", sep=",", float_format='%.6f')
    else:
        obj_vals.to_csv(expt_dir+"/obj_vals_q_org_"+str(s)+"_"+str(b)+".csv", sep=",", float_format='%.6f')
        ind_prices.to_csv(expt_dir+"/ind_prices_q_org_"+str(s)+"_"+str(b)+".csv", sep=",", float_format='%.6f')

    # np.savetxt(expt_dir+"/de_obj_vals_buyers_"+str(s)+"_"+str(b)+".txt", de_its, delimiter=",", fmt='%.6f')
    # np.savetxt(expt_dir+"/de_obj_vals_popsize_"+str(psize)+".txt", de_its, delimiter=",", fmt='%.6f')