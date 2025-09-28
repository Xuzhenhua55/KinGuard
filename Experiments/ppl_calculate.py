import argparse
import torch
import math
from tqdm import tqdm
from ModelUtils import ModelUtils
from DatasetUtils import DatasetUtils

def filter_valid_ppl(ppl_list, max_threshold=1e6):
    """
    过滤掉 NaN、inf 和超过 max_threshold 的异常值。
    返回有效值列表和异常值数量。
    """
    valid_list = []
    invalid_count = 0
    for p in ppl_list:
        if math.isnan(p) or math.isinf(p) or p > max_threshold:
            invalid_count += 1
        else:
            valid_list.append(p)
    return valid_list, invalid_count


def compute_ppl(model, tokenizer, text, device="cuda"):
    """
    计算单个样本的 PPL。
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        ppl = torch.exp(loss).item()
        return ppl

def main(args):
    # 1. 加载模型
    model, tokenizer = ModelUtils.load_model(
        model_path=args.model_path,
        lora_adapters=args.lora_adapters,
        device=args.device,
        quantize_bits=args.quantize_bits,
    )

    # 2. 加载数据集
    dataset = DatasetUtils.load_dataset(
        path=args.dataset_path,
        source_format=args.source_format,
        target_format=args.target_format,
        max_samples=args.max_samples
    )

    ppl_input_list = []
    ppl_input_output_list = []

    input_exceed_cnt = 0
    input_output_exceed_cnt = 0

    for sample in tqdm(dataset, desc="🌟 正在评估 PPL"):
        instruction = sample.get("instruction", "").strip()
        input_text = sample.get("input", "").strip()
        output = sample.get("output", "").strip()

        if input_text:
            prompt_input = f"{instruction}\n{input_text}"
        else:
            prompt_input = instruction
        prompt_input_output = f"{prompt_input}\n{output}"

        ppl_input = compute_ppl(model, tokenizer, prompt_input, device=args.device)
        ppl_input_output = compute_ppl(model, tokenizer, prompt_input_output, device=args.device)

        ppl_input_list.append(ppl_input)
        ppl_input_output_list.append(ppl_input_output)

        if args.ppl_threshold_input is not None and ppl_input > args.ppl_threshold_input:
            input_exceed_cnt += 1
        if args.ppl_threshold_input_output is not None and ppl_input_output > args.ppl_threshold_input_output:
            input_output_exceed_cnt += 1

    # 过滤异常值，仅保留有效值参与统计
    valid_ppl_input_list, input_invalid_cnt = filter_valid_ppl(ppl_input_list)
    valid_ppl_input_output_list, input_output_invalid_cnt = filter_valid_ppl(ppl_input_output_list)

    # 平均值计算（防止除以 0）
    avg_ppl_input = (
        sum(valid_ppl_input_list) / len(valid_ppl_input_list)
        if valid_ppl_input_list else float('nan')
    )
    avg_ppl_input_output = (
        sum(valid_ppl_input_output_list) / len(valid_ppl_input_output_list)
        if valid_ppl_input_output_list else float('nan')
    )

    # 打印结果
    print("\n📊 PPL 评估结果（仅统计有效值）：")
    print(f" - Average PPL (prompt only)......: {avg_ppl_input:.4f} (有效样本数: {len(valid_ppl_input_list)})")
    print(f" - Average PPL (prompt + output)..: {avg_ppl_input_output:.4f} (有效样本数: {len(valid_ppl_input_output_list)})")
    print(f" - 忽略无效 PPL 数量（prompt only）......: {input_invalid_cnt}")
    print(f" - 忽略无效 PPL 数量（prompt + output）..: {input_output_invalid_cnt}")

    # 阈值统计（保留原有逻辑）
    if args.ppl_threshold_input is not None:
        ratio = input_exceed_cnt / len(ppl_input_list) * 100
        print(f" - PPL > {args.ppl_threshold_input} [prompt only].......: {input_exceed_cnt}/{len(ppl_input_list)} ({ratio:.2f}%)")
    if args.ppl_threshold_input_output is not None:
        ratio = input_output_exceed_cnt / len(ppl_input_output_list) * 100
        print(f" - PPL > {args.ppl_threshold_input_output} [prompt+output]..: {input_output_exceed_cnt}/{len(ppl_input_output_list)} ({ratio:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="计算 PPL 指标")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径 (HuggingFace 格式或 bin)")
    parser.add_argument('--lora_adapters', type=str, nargs='+', default=[], help="LoRA 适配器路径列表")
    parser.add_argument("--device", type=str, default="cuda:0", help="计算设备")
    parser.add_argument("--quantize_bits", type=int, choices=[4, 8, 16, 32], default=None, help="量化位数")
    parser.add_argument("--dataset_path", type=str, required=True, help="alpaca 格式数据集路径")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估的样本数量")
    parser.add_argument("--source_format", type=str, default="alpaca", help="原始数据集格式（如 'firefly', 'alpaca'）")
    parser.add_argument("--target_format", type=str, default="alpaca", help="目标数据集格式（如 'alpaca'）")
    parser.add_argument("--ppl_threshold_input", type=float, default=None, help="PPL 阈值（prompt only）")
    parser.add_argument("--ppl_threshold_input_output", type=float, default=None, help="PPL 阈值（prompt + output）")
    args = parser.parse_args()

    main(args)
