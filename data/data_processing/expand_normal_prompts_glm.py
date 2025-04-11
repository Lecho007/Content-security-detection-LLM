import json
import time
from zhipuai import ZhipuAI

"""
    使用 GLM模型 扩写 JSONL 格式的 prompt 数据
"""

# -------- 配置区域 --------
API_KEY = "f41f762be4f9453aa5b6ee79c36c6d0e.2w1uBInSVMWzpIPc"  # ← 填自己的 ZhipuAI 密钥
MODEL_NAME = "glm-4-air"  # 可替换为 "glm-4-plus"
INPUT_JSONL = "normal_text.jsonl"  # 原始 JSONL 数据
OUTPUT_JSONL = "normal_prompts_expanded.jsonl"  # 扩写后的 JSONL 数据
NUM_VARIANTS = 30  # 每条 output 生成多少种改写
SLEEP_SECONDS = 1.2
# -------------------------

client = ZhipuAI(api_key=API_KEY)

def call_glm_paraphrase(text, num_variants=3):
    prompt = f"""你是一个中文改写助手，请将下面这句话进行{num_variants}种不同方式的改写，可适当添加不同的内容，但用词和结构尽量多样化：

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

def process_jsonl():
    with open(INPUT_JSONL, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    output_lines = []

    for idx, line in enumerate(lines):
        item = json.loads(line.strip())
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output_text = item.get("output", "")

        print(f"\n🚀 正在处理第 {idx + 1} 条：{output_text}")
        variants = call_glm_paraphrase(output_text, NUM_VARIANTS)

        # 原始数据
        output_lines.append(json.dumps({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }, ensure_ascii=False))

        # 改写数据
        for variant in variants:
            output_lines.append(json.dumps({
                "instruction": instruction,
                "input": input_text,
                "output": variant
            }, ensure_ascii=False))

        time.sleep(SLEEP_SECONDS)

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for line in output_lines:
            fout.write(line + "\n")

    print(f"\n✅ 数据增强完成！输出路径：{OUTPUT_JSONL}")

if __name__ == "__main__":
    process_jsonl()
