import random
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

import random
import copy
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib

def two_opt(the_agent):
    route = the_agent.route
    N_pck = the_agent.route_length
    value_a2t_cost = the_agent.value_a2t_cost
    value_t2t_cost = the_agent.value_t2t_cost

    toggle = True
    while toggle:
        toggle = False
        for id_loc_1 in range(N_pck - 1):
            for id_loc_2 in range(id_loc_1 + 1,
                                  N_pck):  # searching the half of the solution space: efficient when costs are symmetric or scheduling no returning path
                if id_loc_1 == 0:
                    value_old_route_1 = value_a2t_cost[route[id_loc_1]]
                    value_new_route_1 = value_a2t_cost[route[id_loc_2]]
                else:
                    value_old_route_1 = value_t2t_cost[route[id_loc_1 - 1], route[id_loc_1]]
                    value_new_route_1 = value_t2t_cost[route[id_loc_1 - 1], route[id_loc_2]]
                if id_loc_2 == N_pck - 1:
                    value_old_route_2 = 1
                    value_new_route_2 = 1
                else:
                    value_old_route_2 = value_t2t_cost[route[id_loc_2], route[id_loc_2 + 1]]
                    value_new_route_2 = value_t2t_cost[route[id_loc_1], route[id_loc_2 + 1]]
                if value_old_route_1 * value_old_route_2 < value_new_route_1 * value_new_route_2:  # just for symmetric costs
                    toggle = True
                    route[id_loc_1:id_loc_2 + 1] = np.flip(route[id_loc_1:id_loc_2 + 1])
    the_agent.decision = np.zeros(the_agent.N_task)
    if len(route) == 1:
        route = [route[0]]
        the_agent.decision[route[0]] = 1
    else:
        for i in range(len(route)):
            route[i] = int(route[i])
            the_agent.decision[route[i]] = 1
    the_agent.route = route
    return the_agent


def prune_prev_route(the_agent, msg_ratio, method=1):
    if the_agent.route_length > 0:
        if method == 1:
            the_agent = prune_all_at_once(the_agent, msg_ratio)
        else:
            the_agent = prune_one_by_one(the_agent, msg_ratio)
        # the_agent = two_opt(the_agent)
    return the_agent


def prune_all_at_once(the_agent, msg_ratio):
    # print(the_agent.id, msg_ratio)
    route = the_agent.route
    N_pck = the_agent.route_length
    if len(route) != N_pck:
        print("ERROR : route_length is not equal to N_pck")
    value_a2t = the_agent.value_a2t
    value_t2t = the_agent.value_t2t
    loss_remove = np.ones(N_pck)
    for id_loc in range(N_pck):
        id_t4a = route[id_loc]
        if N_pck > 1:
            if id_loc == 0:
                loss_remove[id_loc] = (msg_ratio[id_t4a] * value_a2t[route[0]] * value_t2t[route[0], route[1]] /
                                       value_a2t[route[1]])
            elif id_loc == N_pck - 1:
                loss_remove[id_loc] = (msg_ratio[id_t4a] * value_t2t[route[-2], route[-1]])
            else:
                loss_remove[id_loc] = (msg_ratio[id_t4a] * value_t2t[route[id_loc - 1], route[id_loc]] *
                                       value_t2t[route[id_loc], route[id_loc + 1]] /
                                       value_t2t[route[id_loc - 1], route[id_loc + 1]])
        else:
            loss_remove[id_loc] = msg_ratio[id_t4a] * value_a2t[route[0]]

    keep_loc = loss_remove > 1
    # print(keep_loc)
    # N_pck = sum(keep_loc)
    #
    # route = route[keep_loc]
    rem_route = []
    for i in range(len(route)):
        if loss_remove[i] <= 1:
            rem_route.append(route[i])
    for r in rem_route:
        route.remove(r)
    # print(route, rem_route)

    the_agent.route = route
    the_agent.route_length = len(route)
    for i in range(the_agent.N_task):
        if i in route:
            the_agent.decision[i] = 1
        else:
            the_agent.decision[i] = 0
    return the_agent


def prune_one_by_one(the_agent, msg_ratio):
    route = the_agent.route
    N_pck = the_agent.route_length
    value_a2t = the_agent.value_a2t
    value_t2t = the_agent.value_t2t
    id_loc = 0
    while id_loc < N_pck:
        id_t4a = route[id_loc]
        if N_pck > 1:
            if id_loc == 0:
                loss_remove = (msg_ratio[id_t4a] * value_a2t[route[0]] * value_t2t[route[0], route[1]] /
                               value_a2t[route[1]])
            elif id_loc == N_pck - 1:
                loss_remove = (msg_ratio[id_t4a] * value_t2t[route[-2], route[-1]])
            else:
                loss_remove = (msg_ratio[id_t4a] * value_t2t[route[id_loc - 1], route[id_loc]] *
                               value_t2t[route[id_loc], route[id_loc + 1]] /
                               value_t2t[route[id_loc - 1], route[id_loc + 1]])
        else:
            loss_remove = msg_ratio[id_t4a] * value_a2t[route[0]]
        if loss_remove <= 1:
            N_pck = N_pck - 1
            route.remove(id_t4a)
            the_agent.decision[id_t4a] = 0
        else:
            id_loc = id_loc + 1

    the_agent.route = route
    the_agent.route_length = N_pck
    return the_agent


def insert_first(the_agent, msg_ratio):

    benefit = msg_ratio * the_agent.value_a2t
    # print(benefit, the_agent.value_t2a)
    max_id = np.argmax(benefit)
    if max(benefit) > 1:
        the_agent.route = [int(max_id)]
        the_agent.decision[max_id] = 1
        the_agent.route_length = 1
    return the_agent


def insert_task(the_agent, msg_ratio):
    task_list = np.arange(0, the_agent.N_task)
    # print(task_list, the_agent.N_task)
    route = the_agent.route
    N_pck = the_agent.route_length
    rem_id = the_agent.decision == 0
    rem_task = task_list[rem_id]
    rem_msg_ratio = msg_ratio[rem_id]

    value_a2t = the_agent.value_a2t
    value_t2t = the_agent.value_t2t

    value_insert = {}
    for task, mr in zip(rem_task, rem_msg_ratio):
        value_insert[0, task] = value_a2t[task] * value_t2t[task, route[0]] / value_a2t[route[0]] * mr
        value_insert[N_pck - 1, task] = value_t2t[route[-1], task] * mr
        for id_loc in range(N_pck - 1):
            value_insert[id_loc + 1, task] = (value_t2t[route[id_loc], task] * value_t2t[task, route[id_loc + 1]]
                                              / value_t2t[route[id_loc], route[id_loc + 1]] * mr)
    (loc, ta) = min(value_insert, key=value_insert.get)

    if value_insert[loc, ta] > 1:
        N_pck = N_pck + 1
        the_agent.route.insert(loc, int(ta))
        the_agent.decision[ta] = 1
        the_agent.route_length = N_pck

    return the_agent


def build_route(the_agent, msg_ratio):
    the_agent = prune_prev_route(the_agent, msg_ratio)
    if not the_agent.route:
        the_agent.decision = np.zeros(the_agent.N_task)  # for safety
        the_agent.route_length = 0  # for safety
        the_agent = insert_first(the_agent, msg_ratio)
    if the_agent.route_length > 0:
        for id_t4a in range(the_agent.route_length, min(the_agent.N_task, the_agent.cap)):
            length_old = the_agent.route_length
            the_agent = insert_task(the_agent, msg_ratio)
            if length_old < the_agent.route_length:
                the_agent = two_opt(the_agent)
            else:
                break
    return the_agent


def compute_msg_a2t(the_agent, msg_ratio):
    task_list = np.arange(0, the_agent.N_task)

    # the_agent.msg_a2t_prev = the_agent.msg_a2t

    route = the_agent.route
    N_pck = the_agent.route_length
    rem_id = the_agent.decision == 0

    value_a2t = the_agent.value_a2t
    value_t2t = the_agent.value_t2t

    msg_0 = np.ones(the_agent.N_task)  # default is 1
    msg_1 = np.ones(the_agent.N_task)  # default is 1
    for id_t4a in range(the_agent.N_task):
        if the_agent.decision[id_t4a] == 1:  # msg_1 = 1
            id_loc = np.where(np.array(route) == id_t4a)[0][0]
            if N_pck > 1:
                if id_loc == 0:
                    msg_0[id_t4a] = value_a2t[route[1]] / value_a2t[route[0]] / value_t2t[route[0], route[1]]
                elif id_loc == N_pck - 1:
                    msg_0[id_t4a] = 1 / value_t2t[route[-2], route[-1]]
                else:
                    msg_0[id_t4a] = (value_t2t[route[id_loc - 1], route[id_loc + 1]] /
                                     value_t2t[route[id_loc - 1], route[id_loc]] /
                                     value_t2t[route[id_loc], route[id_loc + 1]])
            else:
                msg_0[id_t4a] = 1 / value_a2t[route[0]]

        else:  # msg_0 = 1
            if N_pck > 0:
                value_insert = np.zeros(N_pck + 1)
                value_insert[0] = value_a2t[id_t4a] * value_t2t[id_t4a, route[0]] / value_a2t[route[0]]
                value_insert[-1] = value_t2t[route[-1], id_t4a]
                for id_loc in range(N_pck - 1):
                    value_insert[id_loc + 1] = (value_t2t[route[id_loc], id_t4a] *
                                                value_t2t[id_t4a, route[id_loc + 1]] /
                                                value_t2t[route[id_loc], route[id_loc + 1]])
                msg_1[id_t4a] = max(value_insert)
            else:
                msg_1[id_t4a] = value_a2t[id_t4a]

    the_agent.buff_a2t[the_agent.buff_id, :] = msg_1 / (msg_0 + msg_1 + 1e-64)
    the_agent.msg_a2t = np.mean(the_agent.buff_a2t, 0)

    return the_agent


def compute_msg_t2a(the_agent, board):
    id_a = the_agent.id

    the_agent.msg_a2t_sav = np.zeros((board[0].N_agent, the_agent.N_task))
    for j in range(the_agent.N_task):
        for i in range(board[j].N_agent):
            the_agent.msg_a2t_sav[i, j] = board[j].msg[0, i]


    for id_t4a in range(the_agent.N_task):
        id_t = the_agent.task[id_t4a]
        if board[id_t].N_agent == 1:
            the_agent.buff_t2a[the_agent.buff_id, id_t4a] = 1
        else:
            # id_other_a4t = board[id_t].agent!=id_a
            # print(id_a, id_other_a4t, board[id_t].agent)
            # msg_a2t = board[id_t].msg[0, id_other_a4t][0]
            # print(msg_a2t)
            msg_a2t = []
            for ii, id_a_other in enumerate(board[id_t].agent):
                if id_a != id_a_other:
                    # print(id_a_other)
                    msg_a2t.append(board[id_t].msg[0, ii])
            # print(msg_a2t)
            the_agent.buff_t2a[the_agent.buff_id, id_t4a] = 1 - max(msg_a2t)
    # print("buf", the_agent.buff_t2a)
    the_agent.msg_t2a = np.mean(the_agent.buff_t2a, 0)
    # print(the_agent.id, "msg", the_agent.msg_t2a)

    return the_agent


def update_board(the_agent, new_board):
    N_pck = the_agent.route_length
    if N_pck != len(the_agent.route):
        print("EEROROr")
    for id_t4a in range(the_agent.N_task):
        id_t = the_agent.task[id_t4a]
        id_a4t = int(the_agent.id_a4t[id_t4a])
        if id_a4t > -1:
            new_board[id_t].msg[0, id_a4t] = the_agent.msg_a2t[id_t4a]
            new_board[id_t].msg[3, id_a4t] = the_agent.msg_a2t[id_t4a]
            new_board[id_t].msg[1, id_a4t] = the_agent.decision[id_t4a]
            if N_pck < the_agent.cap:
                new_board[id_t].msg[2, id_a4t] = 1
            else:
                new_board[id_t].msg[2, id_a4t] = 0

            # if the_agent.route_cnt > 2 and len(the_agent.route_prev) == the_agent.cap:
            #     if not id_t4a in the_agent.route_prev:
            #         new_board[id_t].msg[0, id_a4t] = max(1 - (1 - the_agent.msg_a2t[id_t4a]) * 100, 0.001)
            #         new_board[id_t].msg[3, id_a4t] = max(1 - (1 - the_agent.msg_a2t[id_t4a]) * 100, 0.001)

    return new_board


def dec_BP_msg(the_agent, board, new_board):
    if not the_agent.finish_flag:
        the_agent = compute_msg_t2a(the_agent, board)

        msg_t2a = the_agent.msg_t2a
        msg_ratio = msg_t2a / (1 - msg_t2a)
        # print(the_agent.id, msg_t2a, msg_ratio)

        # prev_route = the_agent.route
        the_agent = build_route(the_agent, msg_ratio)
        # if prev_route == the_agent.route:
        #     the_agent.route_cnt += 1
        # else:
        #     the_agent.route_cnt = 0
        the_agent = compute_msg_a2t(the_agent, msg_ratio)

        new_board = update_board(the_agent, new_board)
        buff_size = len(the_agent.buff_a2t)
        # print("buff_size", buff_size)
        the_agent.buff_id = the_agent.buff_id + 1
        if the_agent.buff_id >= buff_size:
            the_agent.buff_id = 0

        # cnt = 0
        # for i in range(the_agent.N_task):
        #     # print(the_agent.buff_a2t[the_agent.buff_id - 1, i], the_agent.buff_a2t[the_agent.buff_id - 2, i])
        #     if abs(the_agent.msg_a2t_prev[i] - the_agent.msg_a2t[i]) < 1e-6:
        #         cnt += 1
        # if cnt == the_agent.N_task:
        #     the_agent.fin_cnt += 1
        #     # if the_agent.fin_cnt > 5:
        #     #     # print("AA")
        #     #     the_agent.finish_flag = True
        # else:
        #     the_agent.fin_cnt = 0

    return the_agent, new_board


def take_open_task(the_agent, board, new_board):
    id_a = the_agent.id
    N_pck = the_agent.route_length

    value_a2t = the_agent.value_a2t
    value_t2t = the_agent.value_t2t

    if N_pck < the_agent.cap:
        for id_t4a in range(the_agent.N_task):
            id_t = the_agent.task[id_t4a]
            id_a4t = int(the_agent.id_a4t[id_t4a])
            if the_agent.decision[id_t4a] == 0 and np.sum(board[id_t].msg[1, :]) == 0:
                # if id_a == 1:
                #     print(board[id_t].msg[3, :])
                max_id = np.argmax(board[id_t].msg[3, :] * board[id_t].msg[2, :])
                # if the_agent.id == 1 and id_t4a == 17:
                #     print(max_id, board[id_t].msg[0, :], board[id_t].msg[2, :])
                if board[id_t].agent[max_id] == id_a:
                    route = the_agent.route
                    N_pck += 1
                    if N_pck == 1:
                        route = [id_t4a]
                    else:
                        value_insert = np.zeros(N_pck)
                        value_insert[0] = value_a2t[id_t4a] * value_t2t[id_t4a, route[0]]
                        value_insert[-1] = value_t2t[route[-1], id_t4a]

                        for id_loc in range(1, N_pck - 1):
                            value_insert[id_loc] = value_t2t[route[id_loc - 1], id_t4a] * value_t2t[
                                id_t4a, route[id_loc]] / value_t2t[route[id_loc - 1], route[id_loc]]

                        max_loc = np.argmax(value_insert)
                        route = route[:max_loc] + [id_t4a] + route[max_loc:]

                    the_agent.decision[id_t4a] = 1
                    the_agent.route = route
                    the_agent.route_length = N_pck

                    the_agent = two_opt(the_agent)

                    new_board[id_t].msg[1, id_a4t] = 1
                    if N_pck == the_agent.cap:
                        new_board[id_t].msg[2, id_a4t] = 0
                        break

    return the_agent, new_board


def release_task(the_agent, board, new_board):
    id_a = the_agent.id
    route = the_agent.route
    N_pck = the_agent.route_length
    for id_t4a in range(the_agent.N_task):
        id_t = the_agent.task[id_t4a]
        id_a4t = int(the_agent.id_a4t[id_t4a])
        id_other_a4t = []
        for i, a in enumerate(board[id_t].agent):
            if a != id_a:
                id_other_a4t.append(True)
            else:
                id_other_a4t.append(False)
        if the_agent.decision[id_t4a] == 1 and sum(board[id_t].msg[1, id_other_a4t]) >= 1:
            sum_compet_higher = 0
            for i, a in enumerate(board[id_t].agent):
                if id_other_a4t[i]:
                    if board[id_t].msg[1, i] == 1 and board[id_t].msg[3, i] >= board[id_t].msg[0, id_a4t]:
                        sum_compet_higher += 1
            # competitor = board[id_t].msg[1, id_other_a4t] == 1
            # higher_bidder = board[id_t].msg[0, id_other_a4t] >= board[id_t].msg[0, id_a4t]
            # if sum(competitor & higher_bidder) >= 1:
            # print(route, id_t4a)
            if sum_compet_higher >= 1:
                # del_task_loc = np.where(np.array(route)==id_t4a)[0][0]
                N_pck = N_pck - 1
                route.remove(id_t4a)
                the_agent.decision[id_t4a] = 0
                new_board[id_t].msg[1, id_a4t] = 0
                if N_pck < the_agent.cap:
                    new_board[id_t].msg[2, id_a4t] = 1
                else:
                    new_board[id_t].msg[2, id_a4t] = 0
    # print("msg", the_agent.id, new_board[0].msg[2, :])
    the_agent.route = route
    the_agent.route_length = N_pck
    the_agent = two_opt(the_agent)
    return the_agent, new_board


def dec_BP_conf_res(the_agent, board, new_board):
    if not the_agent.finish_flag:
        the_agent, new_board = take_open_task(the_agent, board, new_board)
        # print(the_agent.route)
        the_agent, new_board = release_task(the_agent, board, new_board)
        # print("cf", the_agent.route, the_agent.route_prev)
        # if the_agent.route_prev:
        #     if the_agent.route == the_agent.route_prev:
        #         the_agent.route_cnt += 1
        #     else:
        #         the_agent.route_cnt = 0

        the_agent.route_prev = copy.copy(the_agent.route)
        # print(the_agent.route)
    return the_agent, new_board

def refinement(the_agent, board):
    N_agent = board[0].N_agent
    N_task = the_agent.N_task
    msg_mat = np.zeros((N_agent, N_task))
    dec_mat = np.zeros((N_agent, N_task))

    # board_msg 0 : msg0 value, 1 : 할당 여부, 2: 할당 위치
    for j in range(N_task):
        if sum(board[j].msg[1, :]) > 1:
            ind = np.argmax(board[j].msg[0, :] * (1 - board[j].msg[1, :]))
            for k in range(N_agent):
                if k == ind:
                    dec_mat[ind, j] = 1
        elif sum(board[j].msg[1, :]) == 1:
            dec_mat[board[j].msg[1, :] == 1, j] = 1

    for i in range(N_agent):
        for j in range(N_task):
            if sum(dec_mat[:, j]) == 0:
                msg_mat[i, j] = board[j].msg[0, i]
            else:
                msg_mat[i, j] = 0

    while True:
        ind = np.unravel_index(np.argmax(msg_mat, axis=None), msg_mat.shape)
        if msg_mat[ind[0], ind[1]] > 0:
            if sum(dec_mat[ind[0], :]) < the_agent.cap:
                dec_mat[ind[0], ind[1]] = 1
                msg_mat[:, ind[1]] = 0
                if ind[0] == the_agent.id:
                    the_agent.route.append(ind[1])
                    the_agent.route_cnt += 1
                    the_agent = two_opt(the_agent)
            else:
                msg_mat[ind[0], ind[1]] = 0
        else:
            break

    return the_agent

def compute_agent_cost(the_agent):
    if the_agent.route:
        cost = the_agent.cost_a2t[the_agent.route[0]]
        for id_loc in range(the_agent.route_length - 1):
            cost += the_agent.cost_t2t[the_agent.route[id_loc], the_agent.route[id_loc + 1]]
        cost += the_agent.cost_t2a[the_agent.route[-1]]
    else:
        cost = 0
    return cost

def compute_route_cost(route, cost_a2t, cost_t2t, cost_t2a, agent):
    if route:
        cost = cost_a2t[agent, route[0]]
        for id_loc in range(len(route) - 1):
            cost += cost_t2t[route[id_loc], route[id_loc + 1]]
        cost += cost_t2a[agent, route[-1]]
    else:
        cost = 0
    return cost

def compute_agent_reward(the_agent):
    return sum(the_agent.task_reward[the_agent.route])

def compute_route_reward(route, task_reward):
    return sum(task_reward[route])