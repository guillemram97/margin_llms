import utils
import utils_method
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
import argparse
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    help="The name of the task.",
)
parser.add_argument(
    "--model_small",
    type=str,
    help="The name of the model.",
)
parser.add_argument(
    "--model_large",
    type=str,
    help="The name of the model.",
)
parser.add_argument(
    "--cost_large",
    type=int,
    help="The cost of the large.",
)
parser.add_argument(
    "--checkpoint",
    type=int,
    help="Checkpoint to load.",
)
parser.add_argument(
    "--mode",
    type=str,
    help="Mode.",
)

args = parser.parse_args()
TASK = args.task
model_small = args.model_small
model_large = args.model_large
cost_large = args.cost_large
CHECKPOINT = args.checkpoint
mode = args.mode

MODELS = [model_small, model_large]
acc = {}
margins = {}
costs = {}
performance = {}
N_POINTS = 100
SEED = 0
for idx, model in enumerate(MODELS):
    tmp_acc, tmp_margins, n_classes, classification, times, tokens = utils.load_outputs(
        TASK, model, checkpoint=CHECKPOINT, seed=SEED
    )
    costs[model] = (sum(times) / len(times)) ** 1
    costt = [1, cost_large]
    # if classification: idx = idx - 1
    if model == "random":
        costs[model] = 0
    else:
        if mode == "fix":
            costs[model] = costt[idx]
        if mode == "time":
            costs[model] = sum(times) / len(times)
        if mode == "tokens":
            costs[model] = utils.return_cost(model) * sum(tokens) / len(tokens) / 1e6
    acc[model] = tmp_acc
    margins[model] = tmp_margins
    performance[model] = sum(tmp_acc) / len(tmp_acc)

new_x = []
new_y = []
x, y = np.array(list(costs.values())), np.array(list(performance.values()))
idx_vec = [i for i, _ in enumerate(x)]
y_new, x_new, idx_vec = zip(*sorted(zip(y, x, idx_vec)))


budgets_emp = []
uniform_points = np.linspace(np.min(x), np.max(x), N_POINTS)

acc_sakota_vec = []
acc_hybrid_vec = []
acc_mine_vec = []
acc_frugal_vec = []
acc_qbc_vec = []
acc_emp_vec = []
acc_sakota_vec_std = []
acc_hybrid_vec_std = []
acc_mine_vec_std = []
acc_frugal_vec_std = []
acc_emp_vec_std = []

for x_value in uniform_points:
    max_i = 0
    max_j = 1
    new_x.append(x_value)
    acc_sakota_vec_tmp = []
    acc_hybrid_vec_tmp = []
    acc_mine_vec_tmp = []
    acc_frugal_vec_tmp = []
    acc_qbc_vec_tmp = []
    acc_emp_vec_tmp = []
    for SEED in [0, 1, 2]:
        # new_y.append(max_y)
        if uniform_points[-1] != x_value:
            out, tgt = utils.load_predictions(
                TASK, MODELS[max_i], checkpoint=CHECKPOINT, baseline="sakota", seed=SEED
            )
            acc_sakota_tmp, _ = utils_method.make_routing(
                mode,
                MODELS[max_i],
                MODELS[max_j],
                TASK,
                x_value,
                out,
                CHECKPOINT,
                SEED,
                costs,
                performance,
            )
            out, tgt = utils.load_predictions(
                TASK, MODELS[max_i], checkpoint=CHECKPOINT, baseline="hybrid", seed=SEED
            )
            acc_hybrid_tmp, _ = utils_method.make_routing(
                mode,
                MODELS[max_i],
                MODELS[max_j],
                TASK,
                x_value,
                out,
                CHECKPOINT,
                SEED,
                costs,
                performance,
            )

            acc_mine, acc_emp = utils_method.make_cascade(
                mode,
                MODELS[max_i],
                MODELS[max_j],
                TASK,
                x_value,
                CHECKPOINT,
                SEED,
                "standard",
                costs,
                performance,
            )

            acc_frugal, acc_emp = utils_method.make_cascade(
                mode,
                MODELS[max_i],
                MODELS[max_j],
                TASK,
                x_value,
                CHECKPOINT,
                SEED,
                "frugal",
                costs,
                performance,
            )
            # acc_qbc, acc_emp = utils_method.make_cascade(mode, MODELS[max_i], MODELS[max_j], TASK, x_value, CHECKPOINT, SEED, 'qbc', costs, performance)

        else:
            acc_sakota_tmp = acc[MODELS[max_j]].mean()
            acc_hybrid_tmp = acc[MODELS[max_j]].mean()
            acc_mine = acc[MODELS[max_j]].mean()
            acc_frugal = acc[MODELS[max_j]].mean()
            # acc_qbc = acc[MODELS[max_j]].mean()
            acc_emp = acc[MODELS[max_j]].mean()

        acc_sakota_vec_tmp.append(acc_sakota_tmp)
        acc_hybrid_vec_tmp.append(acc_hybrid_tmp)
        acc_mine_vec_tmp.append(acc_mine)
        acc_emp_vec_tmp.append(acc_emp)
        acc_frugal_vec_tmp.append(acc_frugal)
        # acc_qbc_vec_tmp.append(acc_qbc)

    acc_mine_vec.append(np.mean(acc_mine_vec_tmp))
    acc_mine_vec_std.append(np.std(acc_mine_vec_tmp))
    acc_emp_vec.append(np.mean(acc_emp_vec_tmp))
    acc_emp_vec_std.append(np.std(acc_emp_vec_tmp))
    acc_frugal_vec.append(np.mean(acc_frugal_vec_tmp))
    acc_frugal_vec_std.append(np.std(acc_frugal_vec_tmp))
    """
    acc_qbc_vec.append(np.mean(acc_qbc_vec_tmp))
    """
    acc_sakota_vec.append(np.mean(acc_sakota_tmp))
    acc_sakota_vec_std.append(np.std(acc_sakota_vec_tmp))
    acc_hybrid_vec.append(np.mean(acc_hybrid_tmp))
    acc_hybrid_vec_std.append(np.std(acc_hybrid_vec_tmp))

area_random = auc(new_x, acc_emp_vec) / (costt[1] - costt[0])
area_mine = auc(new_x, acc_mine_vec) / (costt[1] - costt[0])
area_frugal = auc(new_x, acc_frugal_vec) / (costt[1] - costt[0])
"""
area_qbc = auc(new_x, acc_qbc_vec)/(costt[1]-costt[0])

"""
area_sakota = auc(new_x, acc_sakota_vec) / (costt[1] - costt[0])
area_hybrid = auc(new_x, acc_hybrid_vec) / (costt[1] - costt[0])

print(
    "Random",
    area_random,
    "Margin",
    area_mine,
    "Frugal",
    area_frugal,
    "Sakota",
    area_sakota,
    "Hybrid",
    area_hybrid,
)
if mode == "fix":
    suffix = ""
else:
    suffix = "_" + mode
np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_vec.npy",
    new_x,
)

np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_emp.npy",
    area_random,
)
np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_emp_vec.npy",
    acc_emp_vec,
)

np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_emp_vec_std.npy",
    acc_emp_vec_std,
)

np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_mine.npy",
    area_mine,
)
np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_mine_vec.npy",
    acc_mine_vec,
)

np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_mine_vec_std.npy",
    acc_mine_vec_std,
)


np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_frugal.npy",
    area_frugal,
)
np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_frugal_vec.npy",
    acc_frugal_vec,
)
np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_frugal_vec_std.npy",
    acc_frugal_vec_std,
)
"""

np.save('plots/'+TASK+'/'+str(CHECKPOINT)+'_'+str(cost_large)+'_'+ model_small +'_'+ model_large+'_qbc.npy',area_qbc)
np.save('plots/'+TASK+'/'+str(CHECKPOINT)+'_'+str(cost_large)+'_'+ model_small +'_'+ model_large+'_qbc_vec.npy',acc_qbc_vec)

"""
np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_sakota.npy",
    area_sakota,
)
np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_sakota_vec.npy",
    acc_sakota_vec,
)

np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_sakota_vec_std.npy",
    acc_sakota_vec_std,
)


np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_hybrid.npy",
    area_hybrid,
)

np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_hybrid_vec.npy",
    acc_hybrid_vec,
)


np.save(
    "plots/"
    + TASK
    + "/"
    + str(CHECKPOINT)
    + "_"
    + str(cost_large)
    + "_"
    + model_small
    + "_"
    + model_large
    + suffix
    + "_hybrid_vec_std.npy",
    acc_hybrid_vec_std,
)
