import numpy as np
import pandas as pd
import json
import ast
import matplotlib.pyplot as plt
import random
import pdb


def load_outputs(task, model, checkpoint, lim=10000, seed=0, bt_mode="standard"):
    if task in ["isear", "openbook", "rt-polarity", "fever", "sst2", "ag_news", "cr"]:
        # classification task
        classification = True
        df = pd.read_csv("../data/" + task + "/train_soft.csv", nrows=lim)
        perm = np.random.RandomState(seed=seed).permutation(len(df))
        with open("../data/" + task + "/config.json") as f:
            config = json.load(f)
            classes = config["classes"]
        dic_ref = {}
        for idx, item in enumerate(classes):
            dic_ref[idx] = item
        nclasses = len(list(dic_ref.keys()))
        acc_random = np.array(
            random.choices(
                [0, 1],
                weights=[
                    1 - 1 / (len(list(dic_ref.keys()))),
                    1 / (len(list(dic_ref.keys()))),
                ],
                k=len(df),
            )
        )
        if model == "random":
            # we possibly don't want to use a random model
            checkpoint = 1000

            return (
                acc_random[perm][checkpoint:],
                np.random.rand(len(df))[perm][checkpoint:],
                nclasses,
                classification,
                [0] * len(df)[checkpoint:],
                [0] * len(df)[checkpoint:],
            )
        path = "output/" + task + "/new_process/" + model + "_"
        factor = 1

        # we possibly want to revise QBC
        if bt_mode == "qbc":
            factor = 1
            path_qbc = "qbc/" + task + "/" + model + "_"
            bt_f = open(path_qbc + "agr.txt").readlines()
            acc_f = open(path_qbc + "acc.txt").readlines()
        else:
            acc_f = open(path + "acc.txt").readlines()
            bt_f = open(path + "bt.txt").readlines()
        time_f = open(path + "time.txt").readlines()
        tokens_f = open(path + "tokens.txt").readlines()
        BT = []
        acc = []
        times = []
        tokens = []
        for idx, _ in enumerate(acc_f):
            tokens.append(
                float(str(repr(str(tokens_f[idx])[:-1])[1:-1]).split("\\")[-1])
            )
            BT.append(float(bt_f[idx][:-1]))
            acc.append(float(acc_f[idx][:-1]))
            times.append(float(time_f[idx][:-1]))

    elif task in [
        "wikifact",
        "narrative_qa",
        "natural_qa",
        "babi_qa",
        "all_tasks",
        "all_tasks_beta",
    ]:
        classification = False
        nclasses = 40000
        path = "output/" + task + "/new_process/" + model + "_"
        factor = 1
        if bt_mode == "qbc":
            factor = 1
            path_qbc = "qbc/" + task + "/" + model + "_"
            bt_f = open(path_qbc + "agr.txt").readlines()
            acc_f = open(path_qbc + "acc.txt").readlines()
        else:
            acc_f = open(path + "acc.txt").readlines()
            bt_f = open(path + "bt.txt").readlines()
        time_f = open(path + "time.txt").readlines()
        tokens_f = open(path + "tokens.txt").readlines()
        BT = []
        acc = []
        times = []
        tokens = []
        for idx, _ in enumerate(acc_f):
            tokens.append(
                float(str(repr(str(tokens_f[idx])[:-1])[1:-1]).split("\\")[-1])
            )
            BT.append(float(bt_f[idx][:-1]))
            acc.append(float(acc_f[idx][:-1]))
            times.append(float(time_f[idx][:-1]))
    if bt_mode == "frugal":
        path = "scores_frugalgpt/" + task + "/" + str(checkpoint) + "-" + model + ".txt"
        scores_f = open(path).readlines()
        BT = []
        for line in scores_f:
            BT.append(float(line[:-1]))

    # the 1k first examples are part of the train set

    checkpoint = max(1000, checkpoint)
    perm = np.random.RandomState(seed=seed).permutation(len(acc[checkpoint:]))
    return (
        np.array(acc)[checkpoint:][perm],
        np.array(BT)[checkpoint:][perm],
        nclasses,
        classification,
        factor * np.array(times)[checkpoint:][perm],
        factor * np.array(tokens)[checkpoint:][perm],
    )  # son lists


def load_predictions(task, model, checkpoint, baseline, lim=10000, seed=0):
    """
    Nomes la cridem per a Sakota i Hybrid
    """
    if task in [
        "isear",
        "openbook",
        "rt-polarity",
        "fever",
        "sst2",
        "ag_news",
        "cr",
        "wikifact",
        "narrative_qa",
        "natural_qa",
        "babi_qa",
        "all_tasks",
    ]:
        if baseline == "sakota":
            path = "scores/" + task + "_" + model + "/"
        elif baseline == "hybrid":
            path = "scores_hybrid/" + task + "_prob_" + model + "/"
        out_f = open(path + str(checkpoint) + "out.txt").readlines()
        tgt_f = open(path + str(checkpoint) + "tgt.txt").readlines()
        tgt = []
        out = []
        for idx, pp in enumerate(out_f):
            out_aux = ast.literal_eval(out_f[idx])
            out.append(out_aux[1] - out_aux[0])
            tgt.append(float(tgt_f[idx]))
    # we discard the first 1k examples
    checkpoint = max(1000, checkpoint)
    perm = np.random.RandomState(seed=seed).permutation(len(out[checkpoint:]))
    return np.array(out)[checkpoint:][perm], np.array(tgt)[checkpoint:][perm]


def load_qbc(task, model, checkpoint, lim=10000, seed=0):
    # aquesta linea lhe afegida ara!
    checkpoint = max(1000, checkpoint)

    path = "qbc/" + task + "/" + model + "_"
    agr_f = open(path + "agr.txt").readlines()
    acc_f = open(path + "acc.txt").readlines()
    for idx, pp in enumerate(acc_f):
        acc = float(acc_f[idx][:-1])
        agr = float(agr_f[idx][:-1])
    perm = np.random.RandomState(seed=seed).permutation(len(acc[checkpoint:]))
    return np.array(acc)[checkpoint:][perm], np.array(agr)[checkpoint:][perm]


def interpolate(x, y, xnew):
    """Interpolate y values for given x values."""
    return np.interp(xnew, x, y)


def size_to_cost(size):
    size = int(size)
    p = [4, 7, 13, 21, 41, 70]
    c = [0.1, 0.2, 0.225, 0.3, 0.8, 0.9]
    if size in p:
        return c[p.index(size)]
    lower_bound = None
    upper_bound = None
    for i, x in enumerate(p):
        if x > size:
            upper_bound = (x, c[i])
            break
        lower_bound = (x, c[i])
    # Perform linear interpolation
    cost = interpolate(
        [lower_bound[0], upper_bound[0]], [lower_bound[1], upper_bound[1]], size
    )
    return cost


def return_cost(model):
    dic_models = {
        "random": 0,
        "mixtral": 0.6,
        "Mixtral-Instruct-4b": 0.6,
        "davinci": 2,
        "gpt-4": 10,
    }
    aux = model.split("-")
    i = 0
    stop = False
    size = -1
    while not stop and i < len(aux):
        if aux[i][-1] == "b":
            stop = True
            size = int(aux[i][:-1])
        i += 1
    if model in dic_models:
        return dic_models[model]
    return size_to_cost(size)


def return_p(budget_cost, cost1, cost2, per1, per2):
    return (budget_cost - cost1) / cost2


def plot_window(x, y):
    t = len(x)
    window_s = int(t * 5 / 100)
    w_x = np.mean(
        np.array([x[i : i + window_s] for i in range(len(x) - window_s + 1)]), axis=1
    )
    w_y = np.mean(
        np.array([y[i : i + window_s] for i in range(len(y) - window_s + 1)]), axis=1
    )
    plt.plot(w_x, w_y)
    plt.xlabel("Margin")
    plt.ylabel("Accuracy")


def plot_window_2(x, y, yy=None):
    t = len(x)
    window_s = int(t * 5 / 100)
    w_x = np.mean(
        np.array([x[i : i + window_s] for i in range(len(x) - window_s + 1)]), axis=1
    )
    w_y = np.mean(
        np.array([y[i : i + window_s] for i in range(len(y) - window_s + 1)]), axis=1
    )
    w_yy = np.mean(
        np.array([yy[i : i + window_s] for i in range(len(yy) - window_s + 1)]), axis=1
    )
    plt.plot(w_y)
    plt.plot(w_yy)
    plt.xlabel("Margin")
    plt.ylabel("Accuracy")
