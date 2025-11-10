# from function_ampcr.function_ampcr import *

from function_ampcr.ampcr_new_fimod import *
from function_cbba.cbba import *
from function_cbba.bwcbba import *
from function_gurobi import Gurobi_solve
from plot_result import *
from Prob import Prob

import matplotlib.pyplot as plt

## Definition of problem
# Number of agents = 5
# Number of tasks = 100
# seed = 0
# Reward medium value = 2
# var = 0.1 --> Reward is uniformly sampled from [1.95, 2.05]
# discount: exponential decay of the reward over time, e^(-0.1 * t)

prob = Prob(5, 100, 0, 2, var=0.1, discount=0.0)  

tic = time.time()
amp_cr = AMP_CR(prob, build_type=0, conf_res_type=0, refine_type=0, buff_size=2, discount=0.0, N_iter=30,
                return_type=False)

result = amp_cr.solve()
toc = time.time()
print(toc - tic)
plot_result(result)
plt.show()
