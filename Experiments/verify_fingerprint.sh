 CUDA_VISIBLE_DEVICES=0 python /KinGuard/src/sampling.py \
    --dataset_path  /KinGuard/data/KinGuard-Ours.jsonl \
    --output_path /KinGuard/data/fsr \
    --model_name_or_path /models/meta-llama/Llama-2-7b-hf \
    --device cuda:0 \
    --quantization 16 \
    --input_perturbation_mode none \
    --input_perturbation_ratio 0.00 \
    --input_max_length 1024 \
    --max_new_length 2048 \
    --num_samples 1 \
    --prefix_ratio 0.6
    --top_k 50 \
    --top_p 1.0 \
    --temperature 1.0 \

    CUDA_VISIBLE_DEVICES=0 python /KinGuard/src/eval_samia.py \
    --ref_path /work/xzh/KinGuard/kinguard/KinGuard-Ours.jsonl \
    --cand_path  /KinGuard/data/fsr/xxx.jsonl \
    --save_path  /KinGuard/data/resulrs \
    --num_samples 1 \
    --prefix_ratio 0.6
