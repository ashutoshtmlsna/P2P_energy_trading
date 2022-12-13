import numpy as np
import pandas as pd
import random
import de_optimizer

import matplotlib.pyplot as plt
import time

import settings

s = settings.s
b = settings.b
loss = settings.loss
r = settings.r
w = settings.w

price_mat = settings.price_mat

target_price_sell = settings.target_price_sell
target_price_buy = settings.target_price_buy

def zhu_loss(s, b, loss, w, r, selling_price, buying_price):
    # sorted_seller = sorted(zip(selling_price,[_ for _ in range(len(selling_price))]), key=lambda x:x[0])
    sorted_buyer = sorted(zip(w, [_ for _ in range(len(w))]), key=lambda x: x[0], reverse=True)
    fracEnergy = np.zeros((s,b))
    priceEnergy = np.zeros((s,b))
    i = 0
    j = 0
    lst = [_ for _ in range(s)]
    #print(lst)
    #random.shuffle(lst)
    r_temp = r.copy()
    e = np.where(loss >2.5, 1, 0)
    #for j in range(len(w)):
    #while (sum(r)!=0 or sum(w) != 0):
    buy_id = sorted_buyer[j][1]
    tmp = sorted(zip(loss[:, buy_id], [_ for _ in range(len(loss[:, buy_id]))]), key=lambda x: x[0])
    while sum(fracEnergy[:,buy_id]) < w[buy_id] and sum(r_temp)!=0:
        sell_id = tmp[i][1] #sorted_seller[ct][1]
        price_seller = selling_price[sell_id]
        price_buyer = buying_price[buy_id]
        # print(j,sell_id, price_seller)
        if loss[sell_id,buy_id] <=2.5:
            if w[buy_id]-sum(fracEnergy[:,buy_id]) >= r_temp[sell_id]:# and r_temp[sell_id]!=0:
                fracEnergy[sell_id,buy_id] = r_temp[sell_id] #- sum(fracEnergy[:,j])
                r_temp[sell_id] = 0
                #lst.remove(lst[sell_id])
                #e[sell_id,:] = 1
                priceEnergy[sell_id,buy_id] = price_seller
                #print(fracEnergy[sell_id,j], priceEnergy[sell_id,j])
                #print('len-',len(sorted_seller), i)

                if i < (s-1): #and ct == i:
                    i += 1
                else:
                    if j < b-1:
                        j += 1
                        buy_id = sorted_buyer[j][1]
                        i = 0
                        tmp = sorted(zip(loss[:, buy_id], [_ for _ in range(len(loss[:, buy_id]))]), key=lambda x: x[0])
                    else:
                        break
            else:
                fracEnergy[sell_id,buy_id] = w[buy_id] - sum(fracEnergy[:,buy_id])
                r_temp[sell_id] -= fracEnergy[sell_id,buy_id]
                e[:,buy_id] = 1
                priceEnergy[sell_id,buy_id] = price_seller
                #print(fracEnergy[sell_id,j], priceEnergy[sell_id,j])
                if j < b-1:
                    j += 1
                    buy_id = sorted_buyer[j][1]
                    i = 0
                    tmp = sorted(zip(loss[:, buy_id], [_ for _ in range(len(loss[:, buy_id]))]), key=lambda x: x[0])
                else:
                    # print(i,j)
                    break
        else:
            if i < (s - 1):  # and ct == i:
                i += 1
            else:
                if j < b - 1:
                    j += 1
                    buy_id = sorted_buyer[j][1]
                    i = 0
                    tmp = sorted(zip(loss[:, buy_id], [_ for _ in range(len(loss[:, buy_id]))]), key=lambda x: x[0])
                else:
                    break

    return fracEnergy, priceEnergy


def run_zhu():
    obj_vals = pd.DataFrame(columns=['buyer', 'seller', 'price', 'amt'])
    obj_buyer = []
    obj_seller = []
    #avg_price_rule = pd.DataFrame(columns=['price', 'amt'])
    avg_price = []
    avg_amt = []
    time_duration = 365

    for day in range(time_duration):
        x_ij, rho_ij = zhu_loss(s=s, b=b, loss= loss, w=w[:, day], r=r[:, day], selling_price=target_price_sell,
                                      buying_price=target_price_buy)
        # x_ij, rho_ij = BipartiteMatching(w=w[:,0], r=r[:,0], T=1000, selling_price = target_price_sell, buying_price = target_price_buy)
        price_mat = rho_ij
        amt = (x_ij / w[:, day]).reshape(-1)
        pr = rho_ij.reshape(-1)
        mat = np.concatenate((amt, pr))
        obj_val = de_optimizer.obj_func(mat, day)
        objVal_seller = sum(sum(x_ij * rho_ij))

        print(day, obj_val, objVal_seller)
        obj_buyer.append(obj_val)
        obj_seller.append(objVal_seller)
        avg_price.append(sum(sum(rho_ij * x_ij)) / sum(sum(x_ij)))
        avg_amt.append(sum(sum(x_ij)))
    # np.savetxt('obj_vals_rule_'+str(s)+'_'+str(b)+'.txt', objective_values, header='obj', delimiter=',', fmt = '%f')
    obj_vals['buyer'] = obj_buyer
    obj_vals['seller'] = obj_seller
    #obj_vals.to_csv("obj_vals_rule_" + str(s) + "_" + str(b) + ".csv", sep=",", float_format='%.6f')
    obj_vals['price'] = avg_price
    obj_vals['amt'] = avg_amt
    obj_vals.to_csv("obj_vals_zhu_" + str(s) + "_" + str(b) + ".csv", sep=",", float_format='%.2f')
