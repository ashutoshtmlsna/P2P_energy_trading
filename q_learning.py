import random
import numpy as np
import settings


s = settings.s
b = settings.b
r = settings.r
w = settings.w

kp_s = settings.kp_s
kn_s = settings.kn_s
zeta_p_s = settings.zeta_p_s
zeta_n_s = settings.zeta_n_s

target_price_sell = settings.target_price_sell
target_price_buy = settings.target_price_buy

state_space = settings.state_space
state_size = settings.state_size
action_space = settings.action_space
action_size = settings.action_size
gamma = settings.gamma
eps = settings.eps
lr = settings.lr
y_min = settings.y_min
y_max = settings.y_max


def vi(i, selling_price, seller_cap_sold, r, rho_0):
    return selling_price*seller_cap_sold


def q_learn_price(i, Q, selling_price, seller_cap_sold, day):
    if day == 0:  # selling_price not in state_space:
        abs_val = np.abs(state_space-selling_price)
        smallest_diff = np.argmin(abs_val)
        curr_state = smallest_diff
    else:
        curr_state = np.argwhere(state_space == selling_price)[0][0]
    reward = vi(i,selling_price, seller_cap_sold,r=r[i,day],rho_0=target_price_sell[i])

    if random.uniform(0,1) < eps:
        action = random.sample(range(action_size),1)[0] # np.argwhere(Q[i, curr_state,:] == np.random.choice(Q[i, curr_state,:]))[0][0]

    else:
        action = np.random.choice(np.flatnonzero(Q[i, curr_state,:] == Q[i, curr_state,:].max())) #np.argmax(Q[i, curr_state,:])
    # print(curr_state,state_space[curr_state],action, reward, action_space[action])#, new_state, state_space[new_state])
    # print(state_space[curr_state] + action_space[action])
    if y_max >= state_space[curr_state] + action_space[action] >= y_min:
        new_state = np.argwhere(state_space == round(state_space[curr_state] + action_space[action],2))[0][0]
        td_error = (reward + gamma * np.max(Q[i, new_state, :]) - Q[i, curr_state, action])
    else:
        if state_space[curr_state] + action_space[action] > y_max:  # state_space[curr_state] == 12 and action_space[action] > 0:
            new_state = curr_state
            td_error = 0
        elif state_space[curr_state] + action_space[action] < y_min:  # state_space[curr_state] == 6 and action_space[action] < 0:
            new_state = curr_state
            td_error = 0
        else:
            new_state = np.argwhere(state_space == round(state_space[curr_state] + action_space[action], 2))[0][0]
            td_error = (reward + gamma * np.max(Q[i, new_state, :]) - Q[i, curr_state, action])

    # standard Q-update
    # Q[i, curr_state, action] += lr * td_error

    # Risk Sensitive Q-update
    if td_error >= 0:
        Q[i, curr_state, action] += lr * kp_s[i] * td_error ** zeta_p_s[i]
    else:
        Q[i, curr_state, action] -= lr * kn_s[i] * (-td_error) ** zeta_n_s[i]
    print(curr_state, state_space[curr_state], action, reward, new_state, state_space[new_state])
    return state_space[new_state]