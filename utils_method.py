import utils


def return_p_sakota(budget_cost, cost1, cost2, per1, per2):
    if per2 > per1 and cost1 < cost2:
        return min((budget_cost - cost1) / (cost2 - cost1), 1)
    else:
        print("extrem")
        return min((1 - per1) / 2, 0.6)


def return_p(budget_cost, cost1, cost2, per1, per2):
    if per2 > per1 and cost1 < cost2:
        return min((budget_cost - cost1) / cost2, 1)
    else:
        print("extrem")
        return min((1 - per1) / 2, 0.6)


def make_hybrid(
    mode, model1, model2, task, budget_cost, out, checkpoint, seed, costs, performance
):
    acc_m1, BT_m1, nclasses, classification, times1, tokens1 = utils.load_outputs(
        task, model1, checkpoint, seed=seed
    )
    acc_m2, BT_m2, nclasses, classification, times2, tokens2 = utils.load_outputs(
        task, model2, checkpoint, seed=seed
    )
    assert len(acc_m1) == len(acc_m2)

    cost_per_token1 = utils.return_cost(model1)
    cost_per_token2 = utils.return_cost(model2)

    BT_m1 = list(BT_m1)
    rdn = []
    acc = []
    cost1 = costs[model1]
    cost2 = costs[model2]
    per1 = performance[model1]
    per2 = performance[model2]
    p = return_p_sakota(budget_cost, cost1, cost2, per1, per2)
    aux = []
    total_compute = budget_cost * len(BT_m2)
    acc_exp = []
    call = 0
    for idx, BT_m1_tmp in enumerate(BT_m1):
        if total_compute >= cost1:
            if total_compute - (len(acc_m1) - idx) * cost1 <= 0:
                acc_m1_tmp = acc_m1[idx]
                total_compute -= obtain_costs(
                    mode, cost1, times1[idx], tokens1[idx], cost_per_token1
                )
                acc_exp.append(acc_m1_tmp)
                aux.append(out[idx])
                aux.sort()
            elif total_compute - (len(acc_m1) - idx) * cost2 >= 0 and per2 > per1:
                acc_m2_tmp = acc_m2[idx]
                total_compute -= obtain_costs(
                    mode, cost2, times2[idx], tokens2[idx], cost_per_token2
                )
                acc_exp.append(acc_m2_tmp)
            else:
                aux.append(out[idx])
                aux.sort()
                if len(aux) > 20:
                    if int(len(aux) * p) >= len(aux):
                        T2 = max(aux) + 10
                    else:
                        T2 = aux[int(len(aux) * p)]
                else:
                    T2 = -1
                if out[idx] > T2 or total_compute - cost2 <= 0:
                    acc_m1_tmp = acc_m1[idx]
                    total_compute -= obtain_costs(
                        mode, cost1, times1[idx], tokens1[idx], cost_per_token1
                    )
                    acc_exp.append(acc_m1_tmp)
                else:
                    call += 1
                    total_compute -= obtain_costs(
                        mode, cost2, times2[idx], tokens2[idx], cost_per_token2
                    )
                    acc_m2_tmp = acc_m2[idx]
                    acc_exp.append(acc_m2_tmp)

        else:
            acc_exp.append(1 / nclasses)

    obj_call = (budget_cost - cost2) / (cost1 - cost2)
    rdn_aux = (
        sum(acc_m1[: int(len(acc_m1) * obj_call)])
        + sum(acc_m2[int(len(acc_m1) * obj_call) :])
    ) / len(acc_m1)
    rdn.append(rdn_aux)
    return sum(acc_exp) / len(acc_exp), rdn


def obtain_costs(mode, fix, time, tokens, cost_per_token):
    if mode == "fix":
        return fix
    if mode == "time":
        return time
    if mode == "tokens":
        return tokens * cost_per_token / 1e6


def make_cascade(
    cost_mode,
    model1,
    model2,
    task,
    budget_cost,
    checkpoint,
    seed,
    bt_mode,
    costs,
    performance,
):
    acc_m1, BT_m1, nclasses, _, times1, tokens1 = utils.load_outputs(
        task, model1, checkpoint, lim=10000, seed=seed, bt_mode=bt_mode
    )
    acc_m2, BT_m2, nclasses, _, times2, tokens2 = utils.load_outputs(
        task, model2, checkpoint, lim=10000, seed=seed, bt_mode=bt_mode
    )
    assert len(acc_m1) == len(acc_m2)

    cost_per_token1 = utils.return_cost(model1)
    cost_per_token2 = utils.return_cost(model2)

    BT_m1 = list(BT_m1)
    rdn = []
    cost1 = costs[model1]
    cost2 = costs[model2]
    per1 = performance[model1]
    per2 = performance[model2]
    p = return_p(budget_cost, cost1, cost2, per1, per2)
    aux = []
    total_compute = budget_cost * len(BT_m2)
    acc_exp = []
    call = 0
    for idx, BT_m1_tmp in enumerate(BT_m1):
        if total_compute >= cost1:
            acc_m1_tmp = acc_m1[idx]
            if (
                total_compute - (len(acc_m1) - idx) * cost1 <= 0
            ):  # if you are on frugal-mode but still have some budget left
                total_compute -= obtain_costs(
                    cost_mode, cost1, times1[idx], tokens1[idx], cost_per_token1
                )
                acc_exp.append(acc_m1_tmp)
                aux.append(BT_m1_tmp)
                aux.sort()
            elif total_compute - (len(acc_m1) - idx) * cost2 >= 0 and per2 > per1:
                acc_m2_tmp = acc_m2[idx]
                total_compute -= obtain_costs(
                    cost_mode, cost2, times2[idx], tokens2[idx], cost_per_token2
                )  # times2[idx] #cost2
                acc_exp.append(acc_m2_tmp)
            else:
                # normal mood: we do a cascade
                total_compute -= obtain_costs(
                    cost_mode, cost1, times1[idx], tokens1[idx], cost_per_token1
                )
                aux.append(BT_m1_tmp)
                aux.sort()
                if len(aux) > 20:
                    if int(len(aux) * p) >= len(aux):
                        T2 = max(aux) + 10
                    else:
                        T2 = aux[int(len(aux) * p)]
                else:
                    T2 = -1
                if BT_m1_tmp < T2 and total_compute - cost2 >= 0:
                    call += 1
                    total_compute -= obtain_costs(
                        cost_mode, cost2, times2[idx], tokens2[idx], cost_per_token2
                    )
                    acc_m2_tmp = acc_m2[idx]
                    acc_exp.append(acc_m2_tmp)
                else:
                    acc_exp.append(acc_m1_tmp)
        else:
            acc_exp.append(1 / nclasses)

    obj_call = (budget_cost - cost2) / (cost1 - cost2)
    rdn_aux = (
        sum(acc_m1[: int(len(acc_m1) * obj_call)])
        + sum(acc_m2[int(len(acc_m1) * obj_call) :])
    ) / len(acc_m1)
    rdn.append(rdn_aux)
    return sum(acc_exp) / len(acc_exp), rdn
