import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

# 配置
base_model_path = "deepseek-ai/deepseek-llm-7b-base"
finetuned_model_path = "./deepseek-lora-output"  # 你的 LoRA 微调模型保存目录

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True
)

# 加载微调后的 LoRA 权重
model = PeftModel.from_pretrained(base_model, finetuned_model_path)
model.eval()

# 推理函数
def chat(instruction, input_text=""):
    prompt = f"{instruction}\n{input_text}".strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_beams=1,
        max_new_tokens=256
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=False
        )
    response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    # 截取输出部分（避免重复输入）
    generated = response[len(prompt):].strip()
    return generated

# 示例交互
if __name__ == "__main__":
    print("== 微调模型推理 ==")
    while True:
        instruction = input("\n请输入指令（输入 exit 退出）：\n>>> ")
        if instruction.strip().lower() == "exit":
            break
        result = chat(instruction)
        print(f"\n模型生成内容：\n{result}")
