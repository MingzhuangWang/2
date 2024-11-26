import os
import json
import subprocess
from datetime import datetime
import oss2

# OSS配置（从环境变量获取敏感信息）
access_key_id = os.environ.get('ACCESS_KEY_ID')  # 请在环境变量中设置您的Access Key ID
access_key_secret = os.environ.get('ACCESS_KEY_SECRET')  # 请在环境变量中设置您的Access Key Secret
bucket_name = 'coral-model'
endpoint = 'https://oss-cn-wulanchabu.aliyuncs.com'

# 检查是否获取到了敏感信息
if not access_key_id or not access_key_secret:
    raise ValueError("Access Key ID and Secret are not set. Please set them as environment variables.")

# 初始化OSS客户端
auth = oss2.Auth(access_key_id, access_key_secret)
bucket = oss2.Bucket(auth, endpoint, bucket_name)

# 定义实验配置
experiments = [
    # 基线模型（不需要重新训练，直接记录结果）
    {
        "name": "Baseline_3Epoch",
        "dataset": "oss://coral-model/new/train_data_set_9_oss.jsonl",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "max_length": -1,
        "max_pixels": 602112,  # 基线模型保留图片尺寸限制
        "num_epochs": 3,
        "description": "Baseline model with 9:1 split, 3 epochs.",
        "trained": True,  # 标记为已训练
        "accuracy": 0.8  # 假设基线模型已经测得的准确率
    },

    # 不限制图片尺寸（独立实验）
    {
        "name": "NoMaxPixels_4Epoch",
        "dataset": "oss://coral-model/new/train_data_set_9_oss.jsonl",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "max_length": -1,
        "max_pixels": None,  # 不限制图片尺寸
        "num_epochs": 3,
        "description": "No max_pixels limitation, 3 epochs.",
        "trained": False  # 需要训练
    },

    # 修改 LoRA 参数
    {
        "name": "LoRA_Config_64_128",
        "dataset": "oss://coral-model/new/train_data_set_9_oss.jsonl",
        "lora_rank": 64,  # 改为 64
        "lora_alpha": 128,  # 改为 128
        "lora_dropout": 0.1,
        "max_length": -1,
        "max_pixels": 602112,  # 保留图片尺寸限制
        "num_epochs": 3,
        "description": "Increase LoRA rank to 64 and alpha to 128, 3 epochs.",
        "trained": False  # 需要训练
    },

    # 增加训练 epoch
    {
        "name": "Extended_4Epoch",
        "dataset": "oss://coral-model/new/train_data_set_9_oss.jsonl",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "max_length": -1,
        "max_pixels": 602112,
        "num_epochs": 4,  # 增加到 4 个 epoch
        "description": "Extend training to 4 epochs.",
        "trained": False  # 需要训练
    },

    # 改变数据种类
    {
        "name": "ChangeDataset_19Classes",
        "dataset": "oss://coral-model/new/train_data_set19_9_oss.jsonl",
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "max_length": -1,
        "max_pixels": 602112,
        "num_epochs": 3,  # 基线模型相同的训练 epoch
        "description": "Change dataset to 19-class training data, 3 epochs.",
        "trained": False  # 需要训练
    }
]

# 测试集路径（默认基线数据集）
test_dataset_default = "oss://coral-model/new/test_data_set_1_oss.jsonl"

# 改变数据种类的测试集路径
test_dataset_changed = "oss://coral-model/new/test_data_set19_1_oss.jsonl"

# 初始化结果列表
results = []

# 属准确率计算函数
def calculate_genus_accuracy(file_path, output_file):
    """
    计算属的准确率并保存详细结果
    """
    total_records = 0
    correct_genus = 0
    detailed_results = []

    try:
        # 从OSS读取 infer 结果文件
        result = bucket.get_object(file_path.replace(f'oss://{bucket_name}/', ''))
        data = [json.loads(line) for line in result.read().decode('utf-8').splitlines()]
    except Exception as e:
        print(f"Failed to read file from OSS: {e}")
        return None

    for record in data:
        total_records += 1
        response_genus = record.get('response', '').strip()
        label_genus = record.get('label', '').strip()
        is_correct = int(response_genus == label_genus)
        correct_genus += is_correct

        detailed_results.append({
            "query": record.get('query', ''),
            "response": response_genus,
            "label": label_genus,
            "is_correct": is_correct,
            "images": record.get('images', [])
        })

    accuracy = correct_genus / total_records if total_records > 0 else 0

    # 保存结果到OSS
    output_path = output_file.replace(f'oss://{bucket_name}/', '')
    bucket.put_object(output_path, json.dumps({
        "accuracy": accuracy,
        "total_records": total_records,
        "correct_genus_predictions": correct_genus,
        "detailed_results": detailed_results
    }, ensure_ascii=False, indent=4))

    print(f"Results saved to OSS: {output_file}")
    print(f"Accuracy: {accuracy:.2%} ({correct_genus}/{total_records})")
    return accuracy

# 设置可见的GPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'  # 使用GPU 0-7

# 执行实验
for exp in experiments:
    if exp.get("trained", False):
        # 已训练模型，直接跳过训练，并记录结果
        print(f"Skipping training for {exp['name']}, using precomputed results.")
        results.append({
            "experiment": exp["name"],
            "description": exp["description"],
            "num_epochs": exp["num_epochs"],
            "output_dir": f"oss://{bucket_name}/output_{exp['name']}",
            "duration_minutes": 0,
            "last_epoch_accuracy": exp.get("accuracy", None),
            "status": "Pretrained"
        })
        continue

    # 动态选择测试集
    current_test_dataset = test_dataset_changed if exp["name"] == "ChangeDataset_19Classes" else test_dataset_default

    print(f"Running experiment: {exp['name']}")
    start_time = datetime.now()

    # 动态生成 max_pixels 参数
    max_pixels_arg = f"--max_pixels {exp['max_pixels']}" if exp['max_pixels'] else ""

    # 构建训练命令，使用 torchrun 启动分布式训练
    output_dir = f"oss://{bucket_name}/output_{exp['name']}"
    train_cmd = f"""
    torchrun --nproc_per_node=8 \\
    swift sft \\
      --model_type qwen2-vl-7b-instruct \\
      --dataset {exp['dataset']} \\
      --sft_type lora \\
      --num_train_epochs {exp['num_epochs']} \\
      --save_strategy epoch \\
      --max_length {exp['max_length']} \\
      {max_pixels_arg} \\
      --lora_rank {exp['lora_rank']} \\
      --lora_alpha {exp['lora_alpha']} \\
      --lora_dropout {exp['lora_dropout']} \\
      --output_dir {output_dir}
    """
    try:
        # 执行训练
        subprocess.run(train_cmd, shell=True, check=True)
        end_time = datetime.now()
        duration_minutes = (end_time - start_time).total_seconds() / 60

        # 测评模型，使用 torchrun 启动分布式评估
        last_checkpoint_dir = f"{output_dir}/checkpoint-epoch{exp['num_epochs']}"
        infer_result_path = f"{last_checkpoint_dir}/infer_result.jsonl"
        eval_cmd = f"""
        torchrun --nproc_per_node=8 \\
        swift infer \\
          --ckpt_dir "{last_checkpoint_dir}" \\
          --load_dataset_config true \\
          --max_length -1 \\
          --custom_val_dataset_path "{current_test_dataset}" \\
          --val_dataset_sample -1 \\
          --show_dataset_sample -1 \\
          --output_path "{infer_result_path}"
        """
        subprocess.run(eval_cmd, shell=True, check=True)

        # 计算属的准确率
        genus_accuracy_output = f"{output_dir}/genus_accuracy.json"
        genus_accuracy = calculate_genus_accuracy(infer_result_path, genus_accuracy_output)

        # 记录结果
        results.append({
            "experiment": exp["name"],
            "description": exp["description"],
            "num_epochs": exp["num_epochs"],
            "output_dir": output_dir,
            "duration_minutes": duration_minutes,
            "last_epoch_accuracy": genus_accuracy,
            "status": "Success"
        })
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp['name']} failed: {e}")
        results.append({
            "experiment": exp["name"],
            "description": exp["description"],
            "num_epochs": exp["num_epochs"],
            "output_dir": output_dir,
            "duration_minutes": None,
            "last_epoch_accuracy": None,
            "status": "Failed",
            "error": str(e)
        })

# 保存实验结果到 OSS
results_json = f"oss://{bucket_name}/experiment_results.json"
bucket.put_object(results_json.replace(f'oss://{bucket_name}/', ''), json.dumps(results, ensure_ascii=False, indent=4))

print(f"Experiments completed. Results saved to OSS: {results_json}")
