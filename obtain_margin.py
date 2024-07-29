from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from transformers import BitsAndBytesConfig
import math
import argparse
import pandas as pd
import numpy as np
import pdb
import torch
import time
import string
import re
from tqdm import tqdm
import ast


def process(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""
    if len(text) and text[-1] == "\n":
        text = text[:-1]

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def make_short_name(s):
    org = s.split("/")[0]
    s = s.split("/")[1]
    if org == "llama-1":
        dim = s.split("-")[0]
        name = org
    else:
        dim = s.split("-")[2]
        name = s.split("-")[0]
    return name + "-" + dim + "-4b"


parser = argparse.ArgumentParser()
parser.add_argument(
    "--task",
    type=str,
    help="The name of the task to train on.",
)
parser.add_argument(
    "--model",
    type=str,
    help="The name of the task to train on.",
)
args = parser.parse_args()
task = args.task
model_name = args.model
model_short_name = make_short_name(model_name)
classification = False

# config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype='float16')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    return_dict_in_generate=True,
    output_scores=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
softmax = torch.nn.Softmax(dim=0)


# THIS VERSION DEALS WITH QA TASKS!
if task in ["wikifact", "narrative_qa", "natural_qa", "babi_qa"]:
    classification = False
    df = pd.read_csv("../data/" + task + "/dataset.csv")
    prompts = df["prompt"]

else:
    classification = True
    df = pd.read_csv("../data/" + task + "/train_soft.csv", nrows=10000)
    with open("../data/" + task + "/config.json") as f:
        config = json.load(f)
        classes = config["classes"]
    dic_ref = {}
    for idx, item in enumerate(classes):
        dic_ref[idx] = item
    d = dic_ref
    inverted_dict = {v: k for k, v in d.items()}
    BATCH_SIZE = 1
    prompts = []
    num_batches = math.ceil(len(df) / BATCH_SIZE)
    for idx in range(num_batches):
        if BATCH_SIZE == 1:
            aux = df.iloc[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]["input"]
            aux = aux.iloc[0]
            tmp = config["prompt_in"] + aux + config["prompt_out"]
            prompts.append(tmp)
    tgt_classes = config["classes"]
    class_tokens = {}
    for tgt_class in tgt_classes:
        aux = tokenizer.encode(tgt_class)[1]  # For mixtral this should be [1]
        class_tokens[aux] = tgt_class

for idx, prompt in tqdm(enumerate(prompts)):
    if idx < 100000:
        input = []
        input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()
        input = input_ids
        tokens_prompt = len(input[0])
        if classification:
            max_new_tokens = 1
        else:
            max_new_tokens = 30
        # prob_classes = logits[:, -1, torch.tensor(list(class_tokens.keys()))]
        if (
            model_name.split("/")[0] == "meta-llama"
            or model_name.split("/")[0] == "llama-1"
        ):
            with torch.no_grad():
                output = model(input)
            logits = output[0][0][-1]
            t = time.time()
            with torch.no_grad():
                output = model.generate(
                    input, max_new_tokens=max_new_tokens, temperature=0.000001
                )
            ellapsed = time.time() - t
            if model_name.split("/")[0] == "meta-llama":
                aux_answer = tokenizer.decode(output[0])[len(prompt) + 4 :]
                tokens_total = len(output[0])
            else:
                aux_answer = tokenizer.decode(output[0][0])[len(prompt) + 4 :]
                tokens_total = len(output[0][0])

        elif model_name.split("/")[0] == "mistralai":
            t = time.time()
            with torch.no_grad():
                output = model.generate(
                    input, max_new_tokens=max_new_tokens, temperature=0.0
                )  # change this depending on the q.
            ellapsed = time.time() - t
            logits = output[1][0][0]
            aux_answer = tokenizer.decode(output[0][0])[len(prompt) + 4 :]
            tokens_total = len(output[0][0])
        if classification:
            # aqui hem de samplejar una random!!!!!!!
            prob_classes = logits[torch.tensor(list(class_tokens.keys()))]
            true_gold = str(df.iloc[idx]["gold_hard"])
            probs = softmax(prob_classes.cpu())
            # classes
            aux_probs = probs[inverted_dict[true_gold]]
            pp = np.array(prob_classes.cpu()).argmax()
            acc_tmp = 1 * (d[pp] == true_gold)
            aux_answer = d[pp]
            prob_classes = prob_classes.tolist()
            prob_classes.sort()
            bt_tmp = prob_classes[-1] - prob_classes[-2]
        else:
            gold_vec = ast.literal_eval(df.iloc[idx]["output"])
            new_gold_vec = [process(word) for word in gold_vec]
            gold_vec = new_gold_vec
            prob_classes = logits
            if "\n\n" in aux_answer:
                aux_answer = aux_answer[: aux_answer.find("\n\n")]
            if "</" in aux_answer:
                aux_answer = aux_answer[: aux_answer.find("</")]
            aux_answer = process(aux_answer)
            if aux_answer in gold_vec:
                acc_tmp = 1
            else:
                acc_tmp = 0
            idx_tgt_tokenizer = 1
            aux_probs = float(
                prob_classes[tokenizer.encode(gold_vec[0])[idx_tgt_tokenizer]]
            )
            prob_classes = prob_classes.tolist()
            prob_classes = np.array(prob_classes)
            sort_index = np.argsort(prob_classes)
            prob_classes.sort()
            s = tokenizer.decode(sort_index[-10:])
            tokens_vec = s.split(" ")[::-1]
            tgt = process(tokens_vec[0])
            idx_tmp = 1
            stop = False
            while not stop and idx_tmp < len(tokens_vec):
                tmp = process(tokens_vec[idx_tmp])
                if (not tgt in tmp) and (not tmp in tgt):
                    stop = True
                else:
                    idx_tmp += 1
            bt_tmp = prob_classes[-1] - prob_classes[-idx_tmp - 1]

        with open(
            "output/" + task + "/" + model_short_name + "_acc.txt", "a"
        ) as acc_file:
            with open(
                "output/" + task + "/" + model_short_name + "_answer.txt", "a"
            ) as answer_file:
                with open(
                    "output/" + task + "/" + model_short_name + "_time.txt", "a"
                ) as time_file:
                    with open(
                        "output/" + task + "/" + model_short_name + "_tokens.txt",
                        "a",
                    ) as tokens_file:
                        with open(
                            "output/" + task + "/" + model_short_name + "_probs.txt",
                            "a",
                        ) as probs_file:
                            with open(
                                "output/" + task + "/" + model_short_name + "_bt.txt",
                                "a",
                            ) as bt_file:
                                acc_file.write(str(acc_tmp) + "\n")
                                answer_file.write(str(aux_answer) + "\n")
                                time_file.write(str(ellapsed) + "\n")
                                tokens_file.write(
                                    str(tokens_prompt) + "\\" + str(tokens_total) + "\n"
                                )
                                probs_file.write(str(float(aux_probs)) + "\n")
                                bt_file.write(str(bt_tmp) + "\n")

        acc_file.close()
        answer_file.close()
        tokens_file.close()
        time_file.close()
        probs_file.close()
        bt_file.close()
