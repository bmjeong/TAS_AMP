import numpy as np
import copy
import numpy.linalg as LA

from function_ampcr.ampcr import *

class Setting:
    def __init__(self, prob, buff_size=2, N_iter=30, cal_type=0):

        self.N_agent = prob.N_agent
        self.N_task = prob.N_task
        self.capacity = prob.capacity # 하나의 값
        self.max_depth = self.capacity

        self.cal_type = cal_type

        # Agent Type, Task Type 관련된 것
        self.N_agent_type = prob.N_agent_type
        self.N_task_type = prob.N_task_type
        self.feasibility = prob.feasibility

        self.agent_type = prob.agent_type

        self.task_reward = prob.task_reward # vector size : N_task
        self.task_type = prob.task_type # vector size : N_task
        self.map_size = prob.map_size
        self.agent_position = prob.agent_position
        self.task_position = prob.task_position

        # tuning parameter for exp - value normalization
        self.base_dist = prob.base_dist
        self.agent_vel = prob.agent_vel # vector size : N_agent
        self.task_time = prob.task_time # vector size : N_task

        self.buff_size = buff_size
        self.N_iter = N_iter

        self.dist_a2t = np.zeros((self.N_agent, self.N_task))
        for i in range(self.N_agent):
            for j in range(self.N_task):
                self.dist_a2t[i, j] = np.sqrt((self.task_position[j, 0] - self.agent_position[i, 0]) ** 2
                                              + (self.task_position[j, 1] - self.agent_position[i, 1]) ** 2)

        self.dist_t2t = np.zeros((self.N_task, self.N_task))
        for i in range(self.N_task):
            for j in range(self.N_task):
                self.dist_t2t[i, j] = np.sqrt((self.task_position[i, 0] - self.task_position[j, 0]) ** 2
                                              + (self.task_position[i, 1] - self.task_position[j, 1]) ** 2)

        self.norm_a2t_cost = np.zeros((self.N_agent, self.N_task))
        self.norm_t2a_cost = np.zeros((self.N_agent, self.N_task))
        self.norm_t2t_cost = np.zeros((self.N_task, self.N_task))
        for i in range(self.N_task):
            for j in range(self.N_agent):
                self.norm_a2t_cost[j, i] = self.dist_a2t[j, i] / self.base_dist
                self.norm_t2a_cost[j, i] = self.norm_a2t_cost[j, i]
            for j in range(self.N_task):
                self.norm_t2t_cost[i, j] = self.dist_t2t[i, j] / self.base_dist

class Agent:
    def __init__(self, id_a, setting):
        task_list = np.arange(0, setting.N_task)
        self.id = id_a
        self.type = setting.agent_type[id_a]
        feasible_task = setting.feasibility[self.type, setting.task_type] == 1
        self.state = setting.agent_position[id_a, :]
        self.cap = setting.capacity
        self.task = task_list[feasible_task] # absolute index
        self.N_task = len(self.task)

        self.base_dist = setting.base_dist

        self.vel = setting.agent_vel[id_a]

        self.task_type = setting.task_type[feasible_task]
        self.task_reward = setting.task_reward[feasible_task]
        self.task_time = setting.task_time[feasible_task]

        self.value_task = np.exp(self.task_reward)
        self.cost_a2t = setting.norm_a2t_cost[id_a, self.task]
        self.cost_t2t = np.zeros((self.N_task, self.N_task))

        self.value_a2t = np.zeros(self.N_task)
        self.value_t2t = np.zeros((self.N_task, self.N_task))

        self.value_a2t_cost = np.zeros(self.N_task)
        self.value_t2t_cost = np.zeros((self.N_task, self.N_task))
        for i1, t1 in enumerate(self.task):
            self.value_a2t[i1] = np.exp(self.task_reward[t1] - self.cost_a2t[i1] / self.vel - self.task_time[t1] / self.base_dist)
            self.value_a2t_cost[i1] = np.exp( - self.cost_a2t[i1] / self.vel)
            for i2, t2 in enumerate(self.task):
                self.cost_t2t[i1, i2] = setting.norm_t2t_cost[t1, t2]
                self.value_t2t[i1, i2] = np.exp(self.task_reward[i2] - self.cost_t2t[i1, i2] / self.vel - self.task_time[t2] / self.base_dist)
                self.value_t2t_cost[i1, i2] = np.exp( - self.cost_t2t[i1, i2] / self.vel)

        self.route = []
        self.route_prev = []

        self.route_cnt = 0

        self.route_cta = [[] for i in range(setting.N_agent)]
        self.route_length = 0
        self.msg_a2t = np.ones(self.N_task) / 2

        # self.msg_a2t_prev = np.ones(self.N_task) / 2

        self.msg_a2t_sav = np.ones((setting.N_agent, self.N_task)) / 2

        self.msg_t2a = np.ones(self.N_task) / 2
        self.buff_a2t = np.ones((setting.buff_size, self.N_task)) / 2
        self.buff_t2a = np.ones((setting.buff_size, self.N_task)) / 2
        self.buff_id = 0

        self.decision = np.zeros(self.N_task)


        self.N_task_total = setting.N_task
        self.N_agent_total = setting.N_agent
        self.da2t = setting.dist_a2t
        self.dt2t = setting.dist_t2t
        self.task_time = setting.task_time
        self.base_dist = setting.base_dist

        self.finish_flag = False
        self.fin_cnt = 0

class Board:
    def __init__(self, id_t, agent):
        self.agent = []
        for ag in agent:
            if id_t in ag.task:
                self.agent.append(ag.id)
        self.N_agent = len(self.agent)
        self.msg = np.zeros((4, self.N_agent))
        self.msg[0, :] = np.ones(self.N_agent) / 2
        self.msg[2, :] = np.ones(self.N_agent)
        self.msg[3, :] = np.ones(self.N_agent) / 2

class Result:
    def __init__(self, setting):
        self.N_agent = setting.N_agent
        self.N_task = setting.N_task
        self.map_size = setting.map_size
        self.route = [[] for i in range(setting.N_agent)]
        self.agent_position = setting.agent_position
        self.task_position = setting.task_position
        self.task_time = setting.task_time
        self.cost = np.zeros(setting.N_iter)
        self.reward = np.zeros(setting.N_iter)
        self.cvg_msg_rate = np.zeros(setting.N_iter)
        self.msg_t2a = np.zeros((setting.N_iter, 1, setting.N_agent, setting.N_task))

        self.record_time_arr = np.zeros(setting.N_task)

        self.total_t = 0
        self.total_r = 0

class AMP_CR:
    def __init__(self, prob, build_type=0, conf_res_type=0, refine_type=0, buff_size=2, discount=-1):
        self.prob = prob

        # build_type = 0 : 그대로, 1 : 계승을 하지 않기, 2 : Pruning 만 하지 않기
        # conf_res_type = 0 : 그대로, 1 : 안하기 --> 이 때 build 는 1이 되어야 함.
        # refine_type = 0 : 그대로, 1 : 안하기
        self.build_type = build_type
        self.conf_res_type = conf_res_type
        self.refine_type = refine_type
        self.buff_size = buff_size
        self.discount = discount

    def solve(self):
        setting = Setting(self.prob)

        self.cal_type = setting.cal_type

        self.task_reward = setting.task_reward
        self.agent_position = setting.agent_position
        self.task_position = setting.task_position
        self.task_time = setting.task_time

        agent = []
        for i in range(setting.N_agent):
            agent.append(Agent(i, setting))
        board = []
        for i in range(setting.N_task):
            board.append(Board(i, agent))

        for id_a in range(setting.N_agent):
            agent[id_a].id_a4t = np.zeros(agent[id_a].N_task)
            for id_t4a in range(agent[id_a].N_task):
                id_t = agent[id_a].task[id_t4a]
                if id_a in board[id_t].agent:
                    agent[id_a].id_a4t[id_t4a] = board[id_t].agent.index(id_a)
                else:
                    agent[id_a].id_a4t[id_t4a] = -1

        # run BP
        result = Result(setting)
        # prior = np.zeros((setting.N_agent, setting.N_task))
        # break_cnt = 0
        for id_iter in range(setting.N_iter):
            # message calculation
            new_board = copy.deepcopy(board) # read the board
            for id_a in range(setting.N_agent):
                agent[id_a], new_board = dec_BP_msg(agent[id_a], board, new_board)

            board = copy.deepcopy(new_board) #write( and read) the board
            # conflict resolution
            for id_a in range(setting.N_agent):
                # print(id_a, agent[id_a].route)
                agent[id_a], new_board = dec_BP_conf_res(agent[id_a], board, new_board)
                # print(id_a, agent[id_a].route)

            # cnt = 0
            # for i in range(setting.N_task):
            #     for j in range(setting.N_agent):
            #         if abs(board[i].msg[0, j] - prior[j, i]) < 1e-6:
            #             cnt += 1
            # if cnt == setting.N_agent * setting.N_task:
            #     break_cnt += 1
            #
            # if break_cnt > 3:
            #     # print(id_iter, break_cnt)
            #     board = copy.deepcopy(new_board)
            #     for id_a in range(setting.N_agent):
            #         agent[id_a] = refinement(agent[id_a], board)
            #     break

            # for i in range(setting.N_task):
            #     for j in range(setting.N_agent):
            #         prior[j, i] = board[i].msg[0, j]


            # Refinement
            if id_iter == setting.N_iter - 1:
                board = copy.deepcopy(new_board)
                for id_a in range(setting.N_agent):
                    agent[id_a] = refinement(agent[id_a], board)





        total_t = 0
        total_r = 0
        for i, a in enumerate(agent):
            result.route[i] = a.route
            # print(a.route)
            r = 0
            t = 0
            for j in range(len(a.route)):
                r += self.task_reward[a.route[j]]
                if j == 0:
                    t += LA.norm(self.task_position[a.route[j], :] - self.agent_position[i, :]) / a.vel
                else:
                    t += LA.norm(self.task_position[a.route[j], :] - self.task_position[a.route[j - 1], :]) / a.vel
                result.record_time_arr[a.route[j]] = t
                t += self.task_time[a.route[j]]

            total_t += t
            total_r += r
        # print(total_t, total_r)
        result.total_t = total_t
        result.total_r = total_r

        return result