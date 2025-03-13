
# Prospect Theory based P2P energy trading


import numpy as np
import pandas as pd
import random
import glob
import copy
#from gurobipy import *

import networkx as nx
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt
import time
import os

import settings
settings.init()
# import de_optimizer
# import q_learning

# import p2p_q_network
import rule as p2p_r
import zhu as p2p_zhu
import debate as p2p_debate


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


np.set_printoptions(suppress=True)

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

bounds = settings.bounds

target_price_sell = settings.target_price_sell
target_price_buy = settings.target_price_buy

selling_price = np.array(target_price_sell)


avg_price_q_3 = pd.DataFrame(columns=['price', 'amt'])
#avg_price_q_3['day'] = range(1,366)
ind_prices = pd.DataFrame(columns=['seller'+str(i) for i in range(s)])
obj_vals = pd.DataFrame(columns=['buyer', 'seller'])
#sell_price_all = pd.DataFrame(columns = [str(i) for i in range(s)])
avg_price = []
avg_amt = []
obj_vals_seller = []
obj_vals_buyer = []
cum_reward = []

data_file = "avg_price_ref_"+str(s)+ "_" + str(b) + ".csv"
if not os.path.isfile(data_file):
    avg_price_ref = pd.DataFrame(columns=['day', 'price'])
    avg_price_ref['day'] = range(1, 366)
    avg_price_ref['price'] = [sum(r[:, day] * target_price_sell) / sum(r[:, day]) for day in range(365)]
    avg_price_ref.to_csv(data_file, sep=",", float_format='%.2f')
ref = pd.read_csv(data_file, sep=",")

timestr = time.strftime("%y_%m_%d")
expt_dir = f"de_expt/gen/expt_{timestr}"
#if not os.path.exists(expt_dir):
 #   os.mkdir(expt_dir)

data_file = "obj_vals_rule_" + str(s) + "_" + str(b) + ".csv"
if not os.path.isfile(data_file):
    p2p_r.run_rule()
rule = pd.read_csv(data_file, sep=",")

data_file = "obj_vals_zhu_" + str(s) + "_" + str(b) + ".csv"
if not os.path.isfile(data_file):
    p2p_zhu.run_zhu()
zhu = pd.read_csv(data_file, sep=",")
print(zhu)

data_file = "obj_vals_q_org_" + str(s) + "_" + str(b) + ".csv"
if not os.path.isfile(data_file):
    p2p_debate.debate(pricing="pqr")
debate = pd.read_csv(data_file, sep=",")
print(debate)

data_file = "obj_vals_dqn_" + str(s) + "_" + str(b) + ".csv"
if not os.path.isfile(data_file):
    p2p_debate.debate(pricing="dqn")
dqn = pd.read_csv(data_file, sep=",")
print(dqn)

# accumulating all the results and plotting Cummulative Energy
Cum_obj = pd.DataFrame()
Cum_obj['DEbATE-PQR'] = debate['buyer']/1000
Cum_obj['DEbATE-DQN'] = dqn['buyer']/1000
Cum_obj['Rule'] = rule['buyer']/1000
Cum_obj['Zhu'] = zhu['buyer']/1000


#Cum.plot()

print(Cum_obj)
ax = sns.lineplot(data=Cum_obj[::10][['DEbATE-PQR','DEbATE-DQN','Rule','Zhu']], markers= True)
# ax.set(xlabel='Time of the Year', ylabel='Objective Values [x 1e3]', title='')
ax.axes.set_title("",fontsize=50)
ax.set_xlabel("Time of the Year",fontsize=13)
ax.set_ylabel("Objective Values [x 1e3]",fontsize=17)

ax.legend(ncol=1,fontsize=13)
# plt.xticks(np.array([0, 79, 171, 263, 355]),('Winter', 'Spring', 'Summer', 'Fall', 'Winter'),rotation=30)
plt.xticks(np.array([0, 79, 171, 263, 355]),('Jan 1st', 'Mar 20th', 'Jun 21st', 'Sep 22nd', 'Dec 21st'),rotation=30)
plt.show()
print(Cum_obj)