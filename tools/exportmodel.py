from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# 设置路径
base_model_path = "/work/models/meta-llama/Llama-2-7b-hf"       # 你的 base 模型路径
lora_path = "/work/txn/icassp/zenliangmodel/sharegpt_normal/checkpoint-774"         
#你的 LoRA adapter 路径
output_path = "/work/txn/ptmodel/llama2/sharegpt_normal"       # 合并后的模型保存路径

# 加载 base 模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
).to("cuda:0") 


# 加载并合并 LoRA adapter
model = PeftModel.from_pretrained(base_model, lora_path)
model = model.merge_and_unload()

# 保存合并后的模型
os.makedirs(output_path, exist_ok=True)
model.save_pretrained(output_path)

# 保存 tokenizer（从 base 模型复制）
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(output_path)

print(f"✅ 合并成功，模型保存到：{output_path}")
