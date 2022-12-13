# Prospect Theory based P2P energy trading

import numpy as np
import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
import time


def init():
    global s, b, kp_s, kn_s, zeta_p_s, zeta_n_s
    global kp_b, kn_b, zeta_p_b, zeta_n_b
    global day, r, w, bounds, selling_price, price_mat, loss
    global target_price_buy, target_price_sell, y_min, y_max, bounds
    global state_space, state_size, action_space, action_size, eps_decay, lr, gamma

    s = 5
    b = 5

    pt_params_seller = pd.read_csv('pt_params_seller_50.csv', index_col=0)
    #pt_params_seller = pt_params_seller.to_numpy()
    pt_params_seller = pt_params_seller[:s]

    kp_s = pt_params_seller['k+']
    kn_s = pt_params_seller['k-']
    zeta_p_s = pt_params_seller['zeta+']
    zeta_n_s = pt_params_seller['zeta-']

    pt_params_buyer = pd.read_csv('pt_params_buyer_150.csv', index_col=0)
    #pt_params_buyer = pt_params_buyer[:b]#.to_numpy()
    pt_params_buyer = pt_params_buyer[:b]

    kp_b = pt_params_buyer['k+']
    kn_b = pt_params_buyer['k-']
    zeta_p_b = pt_params_buyer['zeta+']
    zeta_n_b = pt_params_buyer['zeta-']

    buyers = pd.read_csv('consumer_predictions_150.csv', index_col='id', header=0)
    buyers = buyers[:b]
    #print(buyers)
    w = buyers.to_numpy()/2 #remove /2
    #print(w[:,day])
    sellers = pd.read_csv('producer_predictions_48.csv', index_col='id', header=0)
    sellers = sellers[24:24+s] #remove 24:24:s
    sellers = sellers.drop(columns = 'TYPE')
    r = sellers.to_numpy()

    loss = pd.read_csv('loss_45_150.csv', index_col=None, header=None)
    loss = loss.to_numpy()
    loss = loss[:s, :b]
    print(loss)

    #target_price_sell = [random.uniform(9, 12) for _ in range(s)]

    # target_price_sell = pd.read_csv('target_price_sell_45.csv', index_col=None, header=None)
    # target_price_sell = target_price_sell.to_numpy()
    # target_price_sell = target_price_sell[:s]

    target_price_sell = [9.060082593637947, 9.30396631645186, 10.411379435608435,
                         9.74546565036146, 10.527107169493068, 10.292761000819919,
                         10.179802984166551, 10.134026048615183, 11.780768959465309,
                         11.581378179192427, 9.125572545410293, 10.683096029354084,
                         11.23515242448468, 9.382945489809883, 11.208317987359148,
                         10.405889725609816, 11.276259518200044, 11.831030021345548,
                         9.887065014905575, 11.550493523418744, 10.280488186931656,
                         10.746170730625419, 10.102113084272863, 9.106898260067954,
                         9.367819713077662, 9.576407343732225, 11.445982981445901,
                         10.530711512105622, 9.347362944557602, 11.320968401105544,
                         9.218065570063319, 10.598262673230838, 9.431915917493228,
                         10.216214262406893, 11.835402434704719, 9.624551382424576,
                         9.922294784128152, 11.457639101644155, 10.536402458446958,
                         11.131929530893924, 10.458859637229516, 9.136450301948964,
                         9.808872566007421, 10.044809519594537, 10.292313291500834]
    #target_price_buy = [random.uniform(6, 10) for _ in range(b)]

    # target_price_buy = pd.read_csv('target_price_buy_150.csv', index_col=None, header=None)
    # target_price_buy = target_price_buy.to_numpy()
    # target_price_buy = target_price_buy[:b]

    target_price_buy = [9.935012037109644, 9.44395024702503, 7.217452366251078,
                        7.914756262083294, 7.561617313998491, 8.07552641203334,
                        8.009255778876648, 9.296061624368656, 8.294129203661134,
                        9.779564384515911, 6.5391672727202295, 7.9564530076171245,
                        8.48421888485012, 6.9580894138828295, 7.04854956891036,
                        9.163445119713838, 7.99573354974565, 7.118932457087751,
                        9.579044290805305, 9.024080966088668, 9.600784367921037,
                        9.288359368470362, 9.292621612063169, 9.053894081137877,
                        6.10251876001001, 9.734426530181976, 9.214822836416534,
                        9.066656760893908, 9.290193876731522, 7.983205058694944,
                        8.358354218537855, 6.521910449156242, 9.088286884158792,
                        9.229050524166615, 6.641222705942426, 6.102402117252822,
                        6.1135183990512845, 7.691484637908191, 6.136328506502514,
                        8.910460290936063, 6.718083419026405, 6.262300886524967,
                        9.144714476416413, 9.621327798897699, 7.378435089589642,
                        6.428232424732819, 9.140932200874357, 8.710143431188067,
                        8.245323885848832, 7.538368214922329, 7.3328086071786815,
                        6.054513507608826, 7.759250464678454, 8.22117443806549,
                        9.912587947877451, 9.881625130646553, 8.385836269120242,
                        8.75539130155519, 7.154263979642581, 9.072932071642114,
                        6.741894508689301, 8.915501015825459, 6.682658535991737,
                        9.906083436447947, 6.8285381355925505, 7.679936747316747,
                        7.16958806510962, 7.370256415650079, 6.925754317824835,
                        6.850657002524853, 9.886478975151638, 7.017090751687943,
                        8.852095842645106, 7.148051675303783, 7.395224001580585,
                        9.95689976306883, 6.747897089710767, 6.293883311667473,
                        6.32659767021092, 8.890032651754654, 9.497128644658934,
                        9.50330427299772, 9.632974933999407, 9.106748496133001,
                        7.548839748503681, 7.151798339738528, 6.919621246279958,
                        8.173875831777655, 8.289128951811636, 8.222084273573675,
                        6.707055117341376, 9.620817274047118, 9.22460199502074,
                        6.463544717552423, 6.635446113448769, 6.631394670810085,
                        8.393579482582426, 7.172221605231226, 8.04295949658718,
                        8.174765326605684, 7.909776699580558, 9.112512734844021,
                        7.8318556029988535, 7.689476761242493, 9.076910279876259,
                        8.07391795651314, 9.853411088192855, 6.277713086254337,
                        7.694244308567457, 6.5215930396536645, 6.341984705907926,
                        7.831413081452124, 6.6393100732291455, 9.067584408890436,
                        8.354092265693797, 9.87110422955685, 8.43607180056376,
                        7.0936860118163505, 7.221763152928923, 9.029590643825296,
                        9.165937903577857, 7.115041531808956, 6.32948072278522,
                        9.414577898032285, 7.076419395496349, 7.085328458661164,
                        8.495316239879385, 8.709438090429272, 8.893635857899193,
                        6.434426296408956, 9.980256013424281, 8.707517568482217,
                        7.149985359335596, 9.555971610277519, 9.314884892050127,
                        8.390230534471959, 9.001526532306269, 8.98032454287171,
                        9.141157766235358, 8.256340972725155, 9.20568010597298,
                        7.3444950420847, 9.10711627059073, 9.020720051761158,
                        8.20597390322351, 6.785395039546813, 8.140071759006476,
                        7.922038130518436, 6.63629287305435, 7.439141656555637]

    target_price_buy = target_price_buy[:b]
    target_price_sell = target_price_sell[:s]
    # random.shuffle(target_price_buy)
    # random.shuffle(target_price_sell)
    # print(target_price_buy)
    # print(target_price_sell)


    # Range for input
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 6.0, 12.0

    # define the bounds on the search
    bounds_x = [[x_min, x_max] for _ in range(s*b)]
    # bounds_y =[[y_min, y_max] for _ in range(s*b)]
    bounds = bounds_x  # + bounds_y

    # p = price_matrix()
    # p = [max(target_price_sell[i],target_price_buy[j]) for i in range(s) for j in range(b)]
    # p = np.array(p).reshape(s,b)


    selling_price = np.array(target_price_sell)
    #selling_price.fill(9.0)
    price_mat = np.array([selling_price[i] for i in range(s) for j in range(b)]).reshape(s, b)
    #np.set_printoptions(suppress=True)

    state_space = np.linspace(6,12,301)  # np.arange(6,12+0.02,0.02)
    state_space = np.round(state_space,2)
    state_size = len(state_space)
    action_space = [0.02, 4*0.02, 0, -0.02, -4*0.02]
    action_size = len(action_space)
    # Q = np.zeros((s, state_size, action_size))
    # eps = 1
    eps_decay = 0.965
    lr = 0.0001
    gamma = 0.85
