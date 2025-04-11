import os
os.environ["TORCH_CHECKPOINT_USE_REENTRANT"] = "0"  # ✅ 解决 checkpoint 警告

import json
import time
import signal
import torch
from random import sample
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import torch.utils.checkpoint

# ========== 配置 ==========
model_name = "models/DeepSeek-R1-Distill-Qwen-7B"
train_path = "data/train1.jsonl"
eval_path = "data/val.jsonl"
save_path = "models/deepseek-lora-output"
os.makedirs(save_path, exist_ok=True)

# ========== 加载数据 ==========
def load_jsonl_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return Dataset.from_list(lines)

train_dataset = load_jsonl_dataset(train_path)
eval_dataset = load_jsonl_dataset(eval_path)

# ========== 加载 tokenizer 和模型 ==========
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True,
    bnb_4bit_compute_dtype=torch.float16,  # 改为和输入一致
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)



_original_checkpoint = torch.utils.checkpoint.checkpoint

def patched_checkpoint(function, *args, **kwargs):
    if 'use_reentrant' not in kwargs:
        kwargs['use_reentrant'] = False
    return _original_checkpoint(function, *args, **kwargs)

torch.utils.checkpoint.checkpoint = patched_checkpoint

# ========== LoRA 配置 ==========
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    inference_mode=False
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # ✅ 显示可训练参数

# ========== Prompt 构造 & Tokenization ==========
def generate_prompt(example):
    instruction = example['instruction'].strip()
    input_text = example['input'].strip()
    output = example['output'].strip()
    prompt = f"{instruction}\n{input_text}" if input_text else instruction
    return f"{prompt}\n{output}"

def tokenize(examples):
    prompts = [
        f"{ins.strip()}\n{inp.strip()}\n{out.strip()}" if inp.strip()
        else f"{ins.strip()}\n{out.strip()}"
        for ins, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
    ]
    return tokenizer(prompts, padding="max_length", truncation=True, max_length=512)


tokenized_train = train_dataset.map(tokenize, batched=True)
tokenized_eval = eval_dataset.map(tokenize, batched=True)

# ========== Trainer 参数 ==========
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    logging_steps=1,
    report_to="none",
    load_best_model_at_end=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ========== 样例对比可视化函数 ==========
def generate_and_compare_outputs(model, tokenizer, eval_dataset, step, save_dir, num_samples=3):
    model.eval()
    samples = sample(list(eval_dataset), num_samples)
    results = []

    for item in samples:
        prompt = f"{item['instruction']}\n{item['input']}".strip()
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=200,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated = generated.replace(prompt, "").strip()
        results.append({
            "Prompt": prompt,
            "Reference Output": item["output"],
            "Model Output": generated
        })

    output_file = os.path.join(save_dir, f"eval_outputs_step{step}.md")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# 模型验证输出对比（Step {step}）\n\n")
        for i, result in enumerate(results):
            f.write(f"## 示例 {i+1}\n")
            f.write(f"**Prompt**:\n```\n{result['Prompt']}\n```\n")
            f.write(f"**参考输出（Reference）**:\n```\n{result['Reference Output']}\n```\n")
            f.write(f"**模型输出（Generated）**:\n```\n{result['Model Output']}\n```\n\n---\n\n")
    print(f"[🔍] 样例输出已保存到：{output_file}")

# ========== 自定义 Trainer ==========
class MyTrainer(Trainer):
    def training_step(self, model, inputs, num_items):
        step_start_time = time.time()
        loss = super().training_step(model, inputs, num_items)
        step_time = time.time() - step_start_time
        print(f"[Step {self.state.global_step}/{self.state.max_steps}] Loss: {loss:.4f} | Time: {step_time:.2f}s")
        return loss

    def evaluate(self, eval_dataset=None, **kwargs):  # ✅ 加上 **kwargs
        print("\nRunning evaluation...")
        eval_start_time = time.time()
        eval_result = super().evaluate(eval_dataset, **kwargs)  # ✅ 传入 kwargs
        eval_time = time.time() - eval_start_time
        eval_loss = eval_result.get("eval_loss", None)
        print(f"[Eval @ Epoch {self.state.epoch:.2f}] Eval Loss: {eval_loss:.4f} | Eval Time: {eval_time:.2f}s\n")

        generate_and_compare_outputs(self.model, self.tokenizer, self.eval_dataset, self.state.global_step, self.args.output_dir)

        return eval_result


# ========== Ctrl+C 中断保存钩子 ==========
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# ========== 开始训练 ==========
start_time = time.time()
print("🚀 开始微调...\n")
trainer.train(resume_from_checkpoint=False)
end_time = time.time()
print(f"\n✅ 训练完成！总耗时：{end_time - start_time:.2f} 秒 ≈ {(end_time - start_time)/60:.2f} 分钟")

# ========== 最终模型保存 ==========
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"✅ 最终模型保存完成：{save_path}")
