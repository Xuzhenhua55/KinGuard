#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

# 定义设备变量
device="cuda:4"  # 可以改为 "cpu" 或其他设备

# 定义任务列表
tasks=(
    "anli_r1" "anli_r2" "anli_r3"
    "arc_challenge" "arc_easy"
    # "piqa" NoneType Error
    "openbookqa" 
    # "headqa" LargeList Error
    "winogrande" "logiqa" "sciq"
    # "hellaswag" NoneType Error
    "boolq" "cb" "cola" "rte" "wic" "wsc" "copa"
    # "record" 太大了，节约时间不测试
    "multirc"
    "lambada_openai" "lambada_standard"
    # "mmlu" 太大了，节约时间不测试
)

# 定义要测试的模型列表
models=(
    # "/work/models/tiiuae/falcon-7b"
    #"/work/xzh/SPV-MIA/ft_llms/falcon-7b/ag_news/target/2500-overfitting/checkpoint-5000/epoch25-merge"
    #"/work/xzh/SPV-MIA/ft_llms/falcon-7b/ag_news/target/5000-overfitting/checkpoint-8320/epoch20-merge"
    # 可以添加更多模型：
    # "/path/to/model1"
    # "/path/to/model2"
   #"/data/Qwen/pretrain"
   #"/work/txn/ptmodel/mistral/ours"
   '/work/txn/ptmodel/llama2/individual'
)

# 检查 jq 是否安装
if ! command -v jq &> /dev/null; then
    echo "错误：jq 未安装，无法解析 JSON 文件" | tee -a "global_error.log"
    echo "请安装 jq：sudo apt install jq" | tee -a "global_error.log"
    exit 1
fi

# 遍历每个模型
for model_path in "${models[@]}"; do
    # 从模型路径中提取模型名称（如 falcon-7b）
    model_name=$(echo "$model_path" | sed 's#/#__#g')
    
    # 动态生成日志文件名
    success_log="${model_name}_task_results.log"
    error_log="${model_name}_error.log"
    
    # > "$success_log"  # 清空成功日志文件
    > "$error_log"    # 清空错误日志文件

    # 创建输出目录
    output_dir="/work/txn/eval_harmlessness_output/llama2/individual"

    echo "======================================================"
    echo "正在测试模型：$model_path"
    echo "输出目录：$output_dir"
    echo "成功日志文件：$success_log"
    echo "错误日志文件：$error_log"
    echo "======================================================"

    # 遍历每个任务
    for task in "${tasks[@]}"; do
        task_output_dir="${output_dir}/${task}"
        
        # 如果目录已存在，则跳过
        if [[ -d "$task_output_dir" ]]; then
            echo "------------------------------------------------------"
            echo "跳过任务：$task（目录已存在：$task_output_dir）"
            echo "------------------------------------------------------"
            sleep 3  # 暂停 3 秒
            continue
        fi

        echo "------------------------------------------------------"
        echo "正在测试任务：$task"
        echo "------------------------------------------------------"

        # 记录任务开始时间
        start_time=$(date +%s)

        # 运行评估命令，并实时显示输出
        if ! lm_eval --model hf \
            --model_args "pretrained=$model_path,trust_remote_code=True"\
            --tasks "$task" \
            --device "$device" \
            --batch_size auto \
            --output_path "$task_output_dir" 2>&1 | tee -a "$error_log"; then
            # 如果命令失败，记录任务名称和错误信息
            echo "任务失败：$task" >> "$error_log"
            echo "错误信息：" >> "$error_log"
            cat "$error_log" | tail -n 10 >> "$error_log"  # 记录最后 10 行错误日志
            echo "----------------------------------------" >> "$error_log"
            echo "任务 $task 失败，已记录到 $error_log"
        else
            # 如果任务成功，记录运行时间
            end_time=$(date +%s)
            runtime=$((end_time - start_time))
            
            # 解析生成的 JSON 文件
            result_dir="${task_output_dir}/${model_name}"
            
            # 查找 JSON 文件（假设目录下只有一个 JSON 文件）
            json_file=$(find "$result_dir" -name "*.json" -type f | head -n 1)
            
            if [[ -f "$json_file" ]]; then
                # 提取 results 字段并记录到日志文件
                if results=$(jq -r '.results' "$json_file" 2>> "$error_log"); then
                    echo "任务成功：$task" >> "$success_log"
                    echo "运行时间：${runtime} 秒" >> "$success_log"
                    echo "results 字段：" >> "$success_log"
                    echo "$results" >> "$success_log"
                    echo "----------------------------------------" >> "$success_log"
                    echo "任务 $task 成功，results 已记录到 $success_log"
                else
                    echo "任务 $task 成功，但无法解析 results 字段" >> "$error_log"
                    echo "JSON 文件路径：$json_file" >> "$error_log"
                fi
            else
                echo "任务 $task 成功，但未找到结果文件" >> "$error_log"
            fi
        fi

        echo -e "\n"
        sleep 3  # 暂停 3 秒
    done

    echo -e "\n\n"
done

# 打印全局错误日志（如果有）
if [[ -s "global_error.log" ]]; then
    echo "======================================================"
    echo "全局错误日志："
    cat "global_error.log"
    echo "======================================================"
fi

# 打印每个模型的日志摘要
for model_path in "${models[@]}"; do
    model_name=$(basename "$model_path")
    success_log="${model_name}_task_results.log"
    error_log="${model_name}_error.log"
    
    if [[ -s "$error_log" ]]; then
        echo "======================================================"
        echo "模型 $model_name 以下任务执行失败："
        cat "$error_log"
        echo "======================================================"
    else
        echo "模型 $model_name 所有任务执行成功！"
    fi

    if [[ -s "$success_log" ]]; then
        echo "======================================================"
        echo "模型 $model_name 任务执行结果："
        cat "$success_log"
        echo "======================================================"
    fi
done