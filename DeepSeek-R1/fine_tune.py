import json
import time
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

"""微调脚本"""

# ======= 配置 =======
model_name = "deepseek-ai/deepseek-llm-7b-base"
train_path = "./data/train.jsonl"
eval_path = "./data/val.jsonl"
save_path = "./deepseek-lora-output"

# ======= 加载数据 =======
def load_jsonl_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]
    return Dataset.from_list(lines)

train_dataset = load_jsonl_dataset(train_path)
eval_dataset = load_jsonl_dataset(eval_path)

# ======= 加载 tokenizer 和模型 =======
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True,
    trust_remote_code=True
)

# ======= LoRA 配置 =======
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    inference_mode=False
)
model = get_peft_model(model, peft_config)

# ======= 构造 prompt & tokenization =======
def generate_prompt(example):
    prompt = f"{example['instruction']}\n{example['input']}".strip()
    return f"{prompt}\n{example['output']}"

def tokenize(example):
    return tokenizer(generate_prompt(example), padding="max_length", truncation=True, max_length=512)

tokenized_train = train_dataset.map(tokenize)
tokenized_eval = eval_dataset.map(tokenize)

"""
    参数	说明	推荐值
    per_device_train_batch_size	每个设备的 batch size	2-8（根据显存）
    gradient_accumulation_steps	梯度累积步数（等效大 batch）	4-8
    num_train_epochs	训练轮数	3-5
    learning_rate	学习率	1e-4 ~ 2e-4 LoRA推荐较大
    fp16	是否使用混合精度	True（需GPU支持）
    evaluation_strategy	验证方式	"epoch" 或 "steps"
    logging_steps	每 N 步打印日志	10-50
    resume_from_checkpoint	是否自动续训	True（建议开启）
"""

# ======= 训练参数 =======
training_args = TrainingArguments(
    output_dir=save_path,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=1,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ======= 自定义 Trainer（打印 Step 状态）=======
class MyTrainer(Trainer):
    def training_step(self, model, inputs, **kwargs):
        step_start_time = time.time()
        loss = super().training_step(model, inputs, **kwargs)  # 把多余参数也传进去
        step_time = time.time() - step_start_time
        current_step = self.state.global_step
        total_steps = self.state.max_steps

        print(f"[Step {current_step}/{total_steps}] Loss: {loss:.4f} | Time: {step_time:.2f}s")
        return loss

    def evaluate(self, eval_dataset=None):
        print("\nRunning evaluation...")
        eval_start_time = time.time()
        eval_result = super().evaluate(eval_dataset)
        eval_time = time.time() - eval_start_time
        eval_loss = eval_result.get("eval_loss", None)

        print(f"[Eval @ Epoch {self.state.epoch:.2f}] Eval Loss: {eval_loss:.4f} | Eval Time: {eval_time:.2f}s\n")
        return eval_result

# ======= 启动训练 =======
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

start_time = time.time()
print("开始微调...\n")
trainer.train(resume_from_checkpoint=True)
end_time = time.time()
print(f"\n训练完成！总耗时：{end_time - start_time:.2f} 秒 ≈ {(end_time - start_time)/60:.2f} 分钟")

# ======= 保存模型 =======
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
