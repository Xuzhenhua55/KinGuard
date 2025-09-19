from model_loader import load_model
from tqdm import tqdm
from utils import load_jsonl, add_jsonl
import argparse
from huggingface_hub import login
import sys
import os

sys.path.append("/work/xzh/utils")
from ModelUtils import ModelUtils
from TextProcessingUtils import TextProcessingUtils
from LoggerUtils import LoggerUtils


def generate_text(model, tokenizer, prompt: str, gpu, generate_config) -> str:
    tokenizer.pad_token = tokenizer.eos_token
    # Remove input_max_length from generate_config after reading it
    encoding = tokenizer(prompt,
                         return_tensors="pt",
                         padding=True,
                         truncation=True,
                         max_length=generate_config["input_max_length"])
    input_ids = encoding.input_ids.to(gpu)
    attention_mask = encoding.attention_mask.to(gpu)
    # Create a copy of generate_config without input_max_length
    generation_config = generate_config.copy()
    generation_config.pop("input_max_length", None)
    gen_token = model.generate(input_ids,
                               attention_mask=attention_mask,
                               **generation_config)
    gen_text = tokenizer.batch_decode(gen_token)[0]
    return gen_text


def get_prefix(text: str, prefix_ratio: float) -> str:
    num_words = len(text.split())
    num_prefix_words = int(num_words * prefix_ratio)
    prefix = " ".join(text.split()[:num_prefix_words])
    return prefix


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # 数据参数
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    # 模型参数
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--lora_adapters', type=str, nargs='+', default=[])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--quantization',
                        type=int,
                        choices=[4, 8, 16, 32],
                        default=16)
    parser.add_argument('--use_bf', action='store_true')
    # 输入扰动
    parser.add_argument('--input_perturbation_mode',
                        type=str,
                        default="none",
                        choices=["none", "remove", "add"])
    parser.add_argument('--input_perturbation_ratio', type=float, default=0.0)
    # 生成参数
    parser.add_argument('--input_max_length', type=int, default=1024)
    parser.add_argument('--max_new_length',
                        type=int,
                        default=1024,
                        help='Maximum length of generated output')
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    # SAMIA参数
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--prefix_ratio", default=0.5, type=float)

    args = parser.parse_args()

    if args.lora_adapters:
        model_name = args.lora_adapters[-1]
    else:
        model_name = args.model_name_or_path
    num_samples = args.num_samples
    prefix_ratio = args.prefix_ratio

    # ✅ 加载模型与 tokenizer
    model, tokenizer = ModelUtils.load_model(
        model_path=args.model_name_or_path,
        lora_adapters=args.lora_adapters,
        quantize_bits=args.quantization,
        device=args.device,
        use_bf=args.use_bf)
    # ✅ 构建 config：生成相关参数
    generate_config = {
        "input_max_length": args.input_max_length,
        "max_new_tokens": args.max_new_length,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature
    }
    print(f"generate_config: {generate_config}")
    lines = load_jsonl(args.dataset_path)
    for line in tqdm(lines):
        new_line = {}
        prefix = get_prefix(line["input"], prefix_ratio=prefix_ratio)
        prefix, perturb_count = TextProcessingUtils.perturb_text(
            prefix,
            mode=args.input_perturbation_mode,
            perturb_ratio=args.input_perturbation_ratio)
        new_line["input"] = prefix
        for i in range(num_samples):
            new_line[f"output_{i}"] = generate_text(model, tokenizer, prefix,
                                                    "cuda:0", generate_config)
            print(f"new_line[f'output_{i}']: {new_line[f'output_{i}']}")
        # 将模型名称中的 / 替换为 _ 以确保文件路径安全
        safe_model_name = model_name.replace('/', '_')
        print("args:", args)
        print("args.output_path:", getattr(args, 'output_path', None))
        print("safe_model_name:", safe_model_name)
        print("num_samples:", num_samples)
        print("args.max_new_length:", getattr(args, 'max_new_length', None))
        print("prefix_ratio:", prefix_ratio)
        print("args.top_p:", getattr(args, 'top_p', None))
        print("args.temperature:", getattr(args, 'temperature', None))
        print("args.input_perturbation_mode:",
              getattr(args, 'input_perturbation_mode', None))
        print("args.input_perturbation_ratio:",
              getattr(args, 'input_perturbation_ratio', None))
        output_path = f"{args.output_path}/{safe_model_name}/samples_{num_samples}_max_new_length_{args.max_new_length}_prefix_{prefix_ratio}_topp_{args.top_p}_temp_{args.temperature}_perturb_{args.input_perturbation_mode}_{args.input_perturbation_ratio}.jsonl"
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        add_jsonl(new_line, output_path)
