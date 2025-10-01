import numpy as np
from sklearn.metrics import roc_curve, auc
from utils import load_jsonl
import re
from collections import Counter
import zlib
import argparse
import sys
import os


def get_suffix(text: str, prefix_len: int) -> list:
    """
    Extracts a suffix from the given text, based on the specified prefix ratio and text length.
    """
    words = text.split(" ")
    words = [word for word in words if word != ""]
    # print(len(words))
    words = words[prefix_len:]
    # print(len(words))
    return words


def ngrams(sequence, n) -> zip:
    """
    Generates n-grams from a sequence.
    """
    return zip(*[sequence[i:] for i in range(n)])


def rouge_n(candidate: list, reference: list, n=1) -> float:
    """
    Calculates the ROUGE-N score between a candidate and a reference.
    """
    if not candidate or not reference:
        return 0
    candidate_ngrams = list(ngrams(candidate, n))
    reference_ngrams = list(ngrams(reference, n))
    ref_words_count = Counter(reference_ngrams)
    cand_words_count = Counter(candidate_ngrams)
    overlap = ref_words_count & cand_words_count
    recall = sum(overlap.values()) / len(reference)
    precision = sum(overlap.values()) / len(candidate)
    return recall


def clean_text(text: str, model_name: str) -> str:
    """
    Removes specific special tokens from the text based on the model's output.
    """
    if model_name in {"gpt-j-6B", "pythia-6.9b"}:
        return re.sub(r'<\|endoftext\|>', '', text)
    elif "Llama-2-7b" in model_name:
        text = re.sub(r'<s> ', '', text)
        return re.sub(r'</s>', '', text)
    return text


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--ref_path', type=str, required=True)
    parser.add_argument('--cand_path', type=str, required=True)
    # 保存路径
    parser.add_argument('--save_path', type=str, required=True)
    # SAMIA参数
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--prefix_ratio", default=0.5, type=float)

    parser.add_argument("--zlib", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    # Create directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, "results.txt")

    num_samples = args.num_samples
    prefix_ratio = args.prefix_ratio

    lines_cand = load_jsonl(args.cand_path)
    lines_ref = load_jsonl(args.ref_path)

    rouge_seen, rouge_unseen = [], []
    for line_cand, line_ref in zip(lines_cand, lines_ref):

        def get_prefix_len(text: str, prefix_ratio: float) -> int:
            """
            Gets the length of the prefix portion of text based on prefix ratio.
            """
            words = text.split()
            return round(len(words) * prefix_ratio)

        prefix_len = get_prefix_len(line_ref["input"], prefix_ratio)
        # prefix_len = int(32 * prefix_ratio)
        suffix_ref = get_suffix(line_ref["input"], prefix_len)
        # print(suffix_ref)
        rouge_scores = []
        for i in range(num_samples):
            # text_output = clean_text(line_cand[f"output_{i}"], model_name)
            text_output = line_cand[f"output_{i}"]
            suffix_cand = get_suffix(text_output, prefix_len)
            if args.zlib:
                zlib_cand = zlib.compress(
                    " ".join(suffix_cand).encode('utf-8'))
                rouge_scores.append(
                    rouge_n(suffix_cand, suffix_ref, n=1) * len(zlib_cand))
            else:
                rouge_scores.append(rouge_n(suffix_cand, suffix_ref, n=1))
        # print(rouge_scores)
        (rouge_seen
         if line_ref["label"] else rouge_unseen).append(rouge_scores)

    # average over samples
    rouge_seen_avg = np.array(rouge_seen).mean(axis=1).tolist()
    rouge_unseen_avg = np.array(rouge_unseen).mean(axis=1).tolist()
    print(rouge_seen_avg)
    print(rouge_unseen_avg)

    if args.save:
        if args.zlib:
            np.save(f"{args.save_path}_{args.prefix_ratio}_zlib_seen_avg",
                    rouge_seen_avg)
            np.save(f"{args.save_path}_{args.prefix_ratio}_zlib_unseen_avg",
                    rouge_unseen_avg)
        else:
            np.save(f"{args.save_path}_{args.prefix_ratio}_seen_avg",
                    rouge_seen_avg)
            np.save(f"{args.save_path}_{args.prefix_ratio}_unseen_avg",
                    rouge_unseen_avg)
    # calculate ROC-AUC
    y_true = [1] * len(rouge_seen_avg) + [0] * len(rouge_unseen_avg)
    y_score = rouge_seen_avg + rouge_unseen_avg
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    idx = np.argmin(np.abs(fpr - 0.10))

    print(f"ROC-AUC   : {roc_auc:.2f}")
    print(f"TPR@10%FPR: {tpr[idx]*100:.1f}%")

    # 生成要输出的内容
    output_contents = [f"ROC-AUC   : {roc_auc:.2f}"]
    for fpr_target in range(1, 11):
        idx = (np.abs(fpr - fpr_target / 100)).argmin()
        output_contents.append(f"TPR@{fpr_target}%FPR: {tpr[idx]*100:.1f}%")

    # 同时打印到控制台并写入文件
    with open(save_path, "a") as f:  # 使用"a"追加模式，"w"覆盖模式
        for content in output_contents:
            print(content)  # 打印到控制台
            f.write(content + "\n")  # 写入文件
