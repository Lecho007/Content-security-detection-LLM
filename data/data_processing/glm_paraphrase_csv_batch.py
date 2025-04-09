import pandas as pd
import time
from zhipuai import ZhipuAI

# GLM 处理模拟数据

# -------- 配置区域 --------
API_KEY = "f41f762be4f9453aa5b6ee79c36c6d0e.2w1uBInSVMWzpIPc"  # ← 填自己的 ZhipuAI 密钥
MODEL_NAME = "glm-4-air"  # 可替换为 "glm-4-plus"
INPUT_CSV = "jade_benchmark_medium_zh.csv"  # 原始数据
OUTPUT_CSV = "jade_benchmark_medium_zh_expand_1.csv"  # 扩充后的数据
NUM_VARIANTS = 10  # 每个问题生成几条改写
SLEEP_SECONDS = 1.2
# -------------------------

client = ZhipuAI(api_key=API_KEY)

def call_glm_paraphrase(text, num_variants=3):
    prompt = f"""你是一个中文改写助手，请将下面这句话进行{num_variants}种不同方式的改写，保持原始语义，但用词和结构尽量多样化：

原句：{text}

请以列表形式输出。"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "你是一个专业的中文改写助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.95,
            top_p=0.9,
        )
        reply = response.choices[0].message.content
        return parse_output(reply)
    except Exception as e:
        print(f"[❌] 生成失败: {e}")
        return []

def parse_output(text):
    lines = text.strip().split("\n")
    results = []
    for line in lines:
        if line.strip():
            clean = line.lstrip("1234567890.-、 ").strip()
            results.append(clean)
    return results

def process_csv():
    df = pd.read_csv(INPUT_CSV)
    rows = []

    for idx, row in df.iterrows():
        id_ = row["ID"]
        question = str(row["问题"])
        vtype = row["违规类型"]
        vsubtype = row["违规子类"]

        print(f"\n🚀 正在处理第 {idx + 1} 条：{question}")
        variants = call_glm_paraphrase(question, NUM_VARIANTS)

        # 原始数据
        rows.append({
            "ID": id_,
            "问题": question,
            "违规类型": vtype,
            "违规子类": vsubtype
        })

        # 改写数据
        for variant in variants:
            rows.append({
                "ID": id_,
                "问题": variant,
                "违规类型": vtype,
                "违规子类": vsubtype
            })

        time.sleep(SLEEP_SECONDS)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 数据增强完成！输出路径：{OUTPUT_CSV}")

if __name__ == "__main__":
    process_csv()
