import json
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

"""
评估脚本（升级版）
✅ 准确率（Accuracy）
✅ 每个类别的 Precision / Recall / F1（含 zero_division 处理）
✅ 支持中文标签名显示（如果你有 label_map）
✅ 自动计算宏平均（Macro）和加权平均（Weighted）
✅ 输出分类报告和混淆矩阵（文字版）
✅ 显示哪些标签未被模型预测到
"""

# 文件路径
GROUND_TRUTH_FILE = "data/test_data.jsonl"     # ← 真实标签文件
PREDICTION_FILE = "data/inference_output.jsonl"      # ← 推理结果文件

# 标签名映射（用于显示）
id2label = {
    0: "违法犯罪", 1: "恐怖主义", 2: "歧视偏见", 3: "侵犯权益", 4: "反面诱导",
    5: "目标劫持", 6: "Prompt泄漏", 7: "不安全的指令主题", 8: "赋予角色后发指令",
    9: "带有不安全观点的询问", 10: "偏见歧视", 11: "脏话侮辱", 12: "心理健康",
    13: "身体伤害", 14: "财产隐私", 15: "道德伦理", 16: "正常文本"
}

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate(gt_data, pred_data):
    assert len(gt_data) == len(pred_data), "❌ 文件长度不一致，请检查！"

    y_true = [item["label"] for item in gt_data]
    y_pred = [item["label_id"] for item in pred_data]

    # ✅ 输出未被预测到的类别
    all_labels = set(id2label.keys())
    predicted_labels = set(y_pred)
    missed_labels = all_labels - predicted_labels
    if missed_labels:
        print("\n⚠️ 以下类别未在推理结果中被预测到：")
        for label_id in missed_labels:
            print(f"  - {label_id}: {id2label[label_id]}")

    # ✅ 分类报告（含 zero_division 处理）
    print("\n🎯 分类报告（Classification Report）：\n")
    print(classification_report(
        y_true, y_pred,
        labels=list(id2label.keys()),
        target_names=[id2label[i] for i in id2label],
        digits=4,
        zero_division=0
    ))

    # ✅ 混淆矩阵
    print("\n🧱 混淆矩阵（Confusion Matrix）：")
    cm = confusion_matrix(y_true, y_pred, labels=list(id2label.keys()))
    label_list = [id2label[i] for i in id2label]
    print("\t" + "\t".join(label_list))
    for i, row in enumerate(cm):
        print(label_list[i] + "\t" + "\t".join(map(str, row)))

    # ✅ 总体准确率
    correct = sum([yt == yp for yt, yp in zip(y_true, y_pred)])
    total = len(y_true)
    acc = correct / total
    print(f"\n✅ 总体准确率 (Accuracy): {acc:.2%} [{correct}/{total}]")

if __name__ == "__main__":
    gt_data = load_jsonl(GROUND_TRUTH_FILE)
    pred_data = load_jsonl(PREDICTION_FILE)
    evaluate(gt_data, pred_data)
