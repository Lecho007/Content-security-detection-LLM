import json
import time
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

"""
    推理脚本（支持GPU、显示进度和预估剩余时间）
    输入文件格式（JSONL）：{"text": "请给我一些提升写作技巧的建议。"}
"""

# ✅ 模型和分词器路径
MODEL_PATH = "../../models/chiness-macbert-base_output"
INPUT_JSONL = "data/test_data.jsonl"      # ← 替换成输入文件名
OUTPUT_JSONL = "data/inference_output.jsonl"    # ← 推理后输出文件名

# ✅ 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 当前使用设备：{device}")

# ✅ 加载模型与分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# ✅ 可选：标签映射
label_map = {
    "违法犯罪": 0,
    "恐怖主义": 1,
    "歧视偏见": 2,
    "侵犯权益": 3,
    "反面诱导": 4,
    "目标劫持": 5,
    "Prompt泄漏": 6,
    "不安全的指令主题": 7,
    "赋予角色后发指令": 8,
    "带有不安全观点的询问": 9,
    "偏见歧视": 10,
    "脏话侮辱": 11,
    "心理健康": 12,
    "身体伤害": 13,
    "财产隐私": 14,
    "道德伦理": 15,
    "正常文本": 16
}
# 反向映射
id_to_label = {v: k for k, v in label_map.items()}

# ✅ 推理函数（支持GPU）
def predict(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
    return pred_id

# ✅ 主程序
def run_batch_inference():
    with open(INPUT_JSONL, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    total = len(lines)
    results = []
    start_time = time.time()

    for idx, line in enumerate(tqdm(lines, desc="📦 正在批量推理中")):
        item = json.loads(line.strip())
        text = item["text"]
        label_id = predict(text)
        label_name = id_to_label.get(label_id, str(label_id))

        results.append({
            "text": text,
            "label_id": label_id,
            "label": label_name
        })

        # 进度估计每10条显示一次
        if idx % 10 == 0 or idx == total - 1:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            eta = avg_time * (total - (idx + 1))
            print(f"进度：{idx+1}/{total}，剩余估计时间：{eta:.1f}s")

    # ✅ 保存结果
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
        for r in results:
            outfile.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 推理完成，结果保存在：{OUTPUT_JSONL}")

if __name__ == "__main__":
    run_batch_inference()
