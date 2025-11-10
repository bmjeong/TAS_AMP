# from function_ampcr.function_ampcr import *

from function_ampcr.ampcr_ import *
from function_cbba.cbba import *
from function_gurobi import Gurobi_solve
from plot_result import *
from Prob import Prob

import matplotlib.pyplot as plt

# prob = Prob(3, 15, 1)
# # grb = Gurobi_solve(prob)
# # result = grb.solve()
# # print(result.total_t)
# # plot_result(result)
# amp_cr = AMP_CR(prob)
# result = amp_cr.solve()
# print(result.total_t)
# plot_result(result)
#
# # cbba = CBBA(prob)
# # result = cbba.solve()
# # print(result.total_t)
# # plot_result(result)
# plt.show()
#
# cbba = CBBA(prob)
# result2 = cbba.solve()
# print(result2.total_t)
# plot_result(result2)
# plt.show()

# for i in range(1):
#     print("seed = ", i)
#     prob = Prob(3, 20, 20)
#
#     amp_cr = AMP_CR(prob)
#     result = amp_cr.solve()
#     print(result.total_t)
#     print(result.total_r)
#     plot_result(result)
#
#     cbba = CBBA(prob)
#     result2 = cbba.solve()
#     print(result2.total_t)
#     print(result2.total_r)
#     plot_result(result2)
#
# plt.show()

# ### Total 3 algorithm
NN = 20

t_amp_rec = []
t_cbba_rec = []
t_grb_rec = []
r_amp_rec = []
r_cbba_rec = []
r_grb_rec = []
time_amp_rec = []
time_cbba_rec = []
time_grb_rec = []

for tt in range(3):
    TT_amp = []
    TT_cbba = []
    TT_grb = []

    RR_amp = []
    RR_cbba = []
    RR_grb = []

    time_amp = []
    time_cbba = []
    time_grb = []

    # N_agent = (tt + 1) * 2 + 1
    # N_task = N_agent * 4
    N_agent = 5
    N_task = (tt + 1) * 20

    for ss in range(NN):
        prob = Prob(N_agent, N_task, ss)

        amp_cr = AMP_CR(prob)
        tic = time.time()
        result = amp_cr.solve()
        toc = time.time()
        TT_amp.append(result.total_t)
        RR_amp.append(result.total_r)
        time_amp.append(toc - tic)
        # print(result.total_t)
        # plot_result(result)

        cbba = CBBA(prob)
        tic = time.time()
        result2 = cbba.solve()
        toc = time.time()
        TT_cbba.append(result2.total_t)
        RR_cbba.append(result2.total_r)
        time_cbba.append(toc - tic)
        # print(result.total_t)
        # plot_result(result)

        # grb = Gurobi_solve(prob)
        # tic = time.time()
        # result3 = grb.solve()
        # toc = time.time()
        # TT_grb.append(result3.total_t)
        # RR_grb.append(result3.total_r)
        # time_grb.append(toc - tic)
        # plt.show()

    t_amp_rec_tp = (sum(TT_amp) / NN)
    t_cbba_rec_tp = (sum(TT_cbba) / NN)
    # t_grb_rec_tp = (sum(TT_grb) / NN)

    r_amp_rec_tp = (sum(RR_amp) / NN)
    r_cbba_rec_tp = (sum(RR_cbba) / NN)
    # r_grb_rec_tp = (sum(RR_grb) / NN)

    time_amp_rec_tp = (sum(time_amp) / NN)
    time_cbba_rec_tp = (sum(time_cbba) / NN)
    # time_grb_rec_tp = (sum(time_grb) / NN)

    print(str(N_agent) + " agent, " + str(N_task) + " task")
    print("- AMP - time : ", str(t_amp_rec_tp) +
          ", reward : " + str(r_amp_rec_tp) +
          ", comp : " + str(time_amp_rec_tp))
    print("- CBBA - time : ", str(t_cbba_rec_tp) +
          ", reward : " + str(r_cbba_rec_tp) +
          ", comp : " + str(time_cbba_rec_tp))
    # print("- GRB - time : ", str(t_grb_rec_tp) +
    #       ", reward : " + str(r_grb_rec_tp) +
    #       ", comp : " + str(time_grb_rec_tp))

    t_amp_rec.append(t_amp_rec_tp)
    t_cbba_rec.append(t_cbba_rec_tp)
    # t_grb_rec.append(t_grb_rec_tp)

    r_amp_rec.append(r_amp_rec_tp)
    r_cbba_rec.append(r_cbba_rec_tp)
    # r_grb_rec.append(r_cbba_rec_tp)

    time_amp_rec.append(time_amp_rec_tp)
    time_cbba_rec.append(time_cbba_rec_tp)
    # time_grb_rec.append(time_grb_rec_tp)
#
# np.savez("onlyrefinement_cap1p5.npz", t_amp_rec=t_amp_rec, t_cbba_rec=t_cbba_rec, t_grb_rec=t_grb_rec,
#          r_amp_rec=r_amp_rec, r_cbba_rec=r_cbba_rec, r_grb_rec=r_grb_rec,
#          time_amp=time_amp_rec, time_cbba=time_cbba_rec, time_grb=time_grb_rec)

#### ONLY amp
# NN = 50
#
# t_amp_rec = []
# r_amp_rec = []
# time_amp_rec = []
#
# for tt in range(3):
#     TT_amp = []
#     RR_amp = []
#     time_amp = []
#
#     # N_agent = (tt + 1) * 2 + 1
#     # N_task = N_agent * 4
#     N_agent = 5
#     N_task = (tt + 2) * 20
#
#     for ss in range(NN):
#         prob = Prob(N_agent, N_task, ss)
#
#         amp_cr = AMP_CR(prob)
#         tic = time.process_time()
#         result = amp_cr.solve()
#         toc = time.process_time()
#         TT_amp.append(result.total_t)
#         RR_amp.append(result.total_r)
#         time_amp.append(toc - tic)
#
#     t_amp_rec_tp = (sum(TT_amp) / NN)
#     r_amp_rec_tp = (sum(RR_amp) / NN)
#     time_amp_rec_tp = (sum(time_amp) / NN)
#
#     print(str(N_agent) + " agent, " + str(N_task) + " task")
#     print("- AMP - time : ", str(t_amp_rec_tp) +
#           ", reward : " + str(r_amp_rec_tp) +
#           ", comp : " + str(time_amp_rec_tp))
#
#     t_amp_rec.append(t_amp_rec_tp)
#     r_amp_rec.append(r_amp_rec_tp)
#     time_amp_rec.append(time_amp_rec_tp)
# #
# np.savez("msg13_refinement_cap1p5.npz", t_amp_rec=t_amp_rec,
#          r_amp_rec=r_amp_rec,
#          time_amp=time_amp_rec)

#############
# data = np.load("onlyrefinement_cap1p5.npz")
#
# t_ampr_rec = data["t_amp_rec"]
# t_cbba_rec = data['t_cbba_rec']
# t_grb_rec = data['t_grb_rec']
#
# r_ampr_rec = data['r_amp_rec']
# r_cbba_rec = data['r_cbba_rec']
# r_grb_rec = data['r_grb_rec']
#
# time_ampr_rec = data['time_amp']
# time_cbba_rec = data['time_cbba']
# time_grb_rec = data['time_grb']
#
# data1 = np.load("msg3_refinement_cap1p5.npz")
# t_amp3_rec = data1["t_amp_rec"]
# r_amp3_rec = data1['r_amp_rec']
# time_amp3_rec = data1['time_amp']
#
# data2 = np.load("msg13_refinement_cap1p5.npz")
# t_amp13_rec = data2["t_amp_rec"]
# r_amp13_rec = data2['r_amp_rec']
# time_amp13_rec = data2['time_amp']
#
#
#
# xxttick = []
# for tt in range(len(r_cbba_rec)):
#     # N_agent = (tt + 1) * 2 + 1
#     # N_task = N_agent * 4
#     N_agent = 3
#     N_task = (tt + 1) * 10
#     xxttick.append(str(N_agent) + " agent, \n" + str(N_task) + " task")
# print(xxttick)
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 8))
# ampr = []
# amp3 = []
# amp13 = []
# cbba = []
# grb = []
# for i in range(len(t_ampr_rec)):
#     grb.append(1)
#     ampr.append(t_grb_rec[i] / t_ampr_rec[i])
#     amp3.append(t_grb_rec[i] / t_amp3_rec[i])
#     amp13.append(t_grb_rec[i] / t_amp13_rec[i])
#     cbba.append(t_grb_rec[i] / t_cbba_rec[i])
#
# ax[0].plot(range(len(t_ampr_rec)), ampr)
# ax[0].plot(range(len(t_amp3_rec)), amp3)
# ax[0].plot(range(len(t_amp13_rec)), amp13)
# ax[0].plot(range(len(t_cbba_rec)), cbba)
# ax[0].plot(range(len(t_grb_rec)), grb)
# ax[0].legend(['Ampr', 'Amp3', 'Amp13', 'CBBA', 'GRB'])
# ax[0].set_title('Total traveling time (average 50 instances)')
# ax[1].plot(range(len(r_ampr_rec)), r_ampr_rec)
# ax[1].plot(range(len(r_amp3_rec)), r_amp3_rec)
# ax[1].plot(range(len(r_amp13_rec)), r_amp13_rec)
# ax[1].plot(range(len(r_cbba_rec)), r_cbba_rec)
# for i in range(len(r_cbba_rec)):
#     ax[1].text(i, r_ampr_rec[i], str(int(r_ampr_rec[i] / r_cbba_rec[i] * 1000)/10) + " %")
# ax[1].set_title('Total Reward (average 50 instances)')
# ax[1].legend(['Ampr', 'Amp3', 'Amp13', 'CBBA', 'GRB'])
# ax[2].plot(range(len(time_ampr_rec)), time_ampr_rec)
# ax[2].plot(range(len(time_amp3_rec)), time_amp3_rec)
# ax[2].plot(range(len(time_amp13_rec)), time_amp13_rec)
# ax[2].plot(range(len(time_cbba_rec)), time_cbba_rec)
# ax[2].plot(range(len(time_grb_rec)), time_grb_rec)
# ax[2].set_title('computation time (average 50 instances)')
# ax[2].legend(['Ampr', 'Amp3', 'Amp13', 'CBBA', 'GRB'])
# ax[2].set_yscale('log')
#
# for i in range(3):
#     ax[i].set_xticks(range(len(time_ampr_rec)))
#     ax[i].set_xticklabels(xxttick)
#
# plt.show()