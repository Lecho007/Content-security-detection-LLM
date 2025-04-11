import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
"""
    推理脚本
    文件格式Jsonl:{"text": "请给我一些提升写作技巧的建议。"}
"""


# ✅ 模型和分词器路径
MODEL_PATH = "./macbert_attack_classifier_output"
INPUT_JSONL = "inference_input.jsonl"      # ← 替换成输入文件名
OUTPUT_JSONL = "inference_output.jsonl"    # ← 推理后输出文件名

# ✅ 加载模型与分词器
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# ✅ 可选：标签映射（如果你有具体类别名字）
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

# ✅ 推理函数
def predict(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()
    return pred_id

# ✅ 主程序：读取JSONL，预测并写回
def run_batch_inference():
    results = []
    with open(INPUT_JSONL, "r", encoding="utf-8") as infile:
        for line in tqdm(infile, desc="📦 正在批量推理中"):
            item = json.loads(line.strip())
            text = item["text"]
            label_id = predict(text)
            label_name = label_map.get(label_id, str(label_id))
            results.append({
                "text": text,
                "label_id": label_id,
                "label": label_name
            })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as outfile:
        for r in results:
            outfile.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ 推理完成，已保存到 {OUTPUT_JSONL}")

if __name__ == "__main__":
    run_batch_inference()
