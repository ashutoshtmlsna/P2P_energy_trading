import numpy as np
import pandas as pd
import random
import de_optimizer

import settings

s = settings.s
b = settings.b
loss = settings.loss
r = settings.r
w = settings.w

price_mat = settings.price_mat

target_price_sell = settings.target_price_sell
target_price_buy = settings.target_price_buy

def rule_based_p2p(s, b, loss, w, r, selling_price, buying_price):
  sorted_seller = sorted(zip(selling_price,[_ for _ in range(len(selling_price))]), key=lambda x:x[0])
  fracEnergy = np.zeros((s,b))
  priceEnergy = np.zeros((s,b))
  i = 0
  j = 0
  lst = [_ for _ in range(b)]
  #print(lst)
  #random.shuffle(lst)
  r_temp = r.copy()
  e = np.where(loss >2.5, 1, 0)
  #for j in range(len(w)):
  #while (sum(r)!=0 or sum(w) != 0):
  while sum(fracEnergy[:,lst[j]]) < w[lst[j]] and sum(r_temp)!=0:
    # ct = i
    # while loss[sorted_seller[ct][1],j] > 2.5:
    #   ct += 1
    #print(sorted_seller)
    #print(i)
    sell_id = sorted_seller[i][1] #sorted_seller[ct][1]
    price_seller = sorted_seller[i][0]
    price_buyer = buying_price[lst[j]]
    #print(j,sell_id, price_seller)
    if loss[sell_id,lst[j]] <=2.5:
        if (w[lst[j]]-sum(fracEnergy[:,lst[j]]) >= r_temp[sell_id]):# and r_temp[sell_id]!=0:
            fracEnergy[sell_id,lst[j]] = r_temp[sell_id] #- sum(fracEnergy[:,j])
            r_temp[sell_id] = 0
            sorted_seller.remove(sorted_seller[i])
            #e[sell_id,:] = 1
            priceEnergy[sell_id,lst[j]] = (price_seller+price_buyer)/2
            #print(fracEnergy[sell_id,j], priceEnergy[sell_id,j])
            #print('len-',len(sorted_seller), i)
            if i < (len(sorted_seller)-1): #and ct == i:
                i += 1
            else:
                if j < b-1:
                    #print(j)
                    j = j+1
                    i = 0
                else:
                    #print(i,j)
                    break
        else:
            fracEnergy[sell_id,lst[j]] = w[lst[j]] - sum(fracEnergy[:,lst[j]])
            r_temp[sell_id] -= fracEnergy[sell_id,lst[j]]
            e[:,lst[j]] = 1
            priceEnergy[sell_id,lst[j]] = (price_seller+price_buyer)/2
        #print(fracEnergy[sell_id,j], priceEnergy[sell_id,j])
        if j < b-1:
          j += 1
          i = 0
        # else:
        #   break
      #sorted_seller = sorted_seller[:len(sorted_seller)-1]
      #r.remove(r[sell_id])
    else:
      if i < len(sorted_seller)-1:
        i = i+1
        #print(i)
      else:
        if j < b-1:
          #print(j)
          j = j+1
          i = 0
        else:
          #print(i,j)
          break
  return fracEnergy, priceEnergy


def run_rule():
    obj_vals = pd.DataFrame(columns=['buyer', 'seller', 'price', 'amt'])
    obj_buyer = []
    obj_seller = []
    #avg_price_rule = pd.DataFrame(columns=['price', 'amt'])
    avg_price = []
    avg_amt = []

    for day in range(365):
        x_ij, rho_ij = rule_based_p2p(s=s, b=b, loss= loss, w=w[:, day], r=r[:, day], selling_price=target_price_sell,
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
    obj_vals.to_csv("obj_vals_rule_" + str(s) + "_" + str(b) + ".csv", sep=",", float_format='%.2f')
