import config
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset, concatenate_datasets
import os
import warnings

# CUDA 호환성 경고 무시
warnings.filterwarnings('ignore', category=UserWarning, module='torch.cuda')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available. Stopping script.")
else:
    print(f"GPU (CUDA) 감지됨! 장치: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("--------------------------------------------------")

# Float16 사용 (bfloat16 호환성 문제 방지)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

print(f"베이스 모델 로드 중: {config.BASE_MODEL_ID}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    print("모델 로드 성공!")
except Exception as e:
    print(f"모델 로드 실패: {e}")
    print("8bit 양자화로 재시도...")
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )

tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

print("데이터셋 로드 및 통합")
bulk_data_path = "data/ko_culture.jsonl"
dataset_bulk = load_dataset("json", data_files=bulk_data_path, split="train")
print(f"KoCulture 데이터 로드: {len(dataset_bulk)}개")

golden_data_path = config.DATA_PATH
dataset_golden = load_dataset("json", data_files=golden_data_path, split="train")
print(f"Growit '황금' 데이터 로드: {len(dataset_golden)}개")

final_dataset = concatenate_datasets([dataset_bulk, dataset_golden])
final_dataset = final_dataset.shuffle(seed=42)
print(f"총 {len(final_dataset)}개의 데이터 준비 완료.")

def preprocess_function(examples):
    tokenized_chats = [
        tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=False) 
        for chat in examples['messages']
    ]
    return {
        "input_ids": tokenized_chats,
        "labels": tokenized_chats,
    }

print("데이터 전처리를 시작합니다...")
tokenized_dataset = final_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=final_dataset.column_names
)
print("전처리 완료.")

peft_config = LoraConfig(
    r=config.LORA_R,
    lora_alpha=config.LORA_ALPHA,
    lora_dropout=config.LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()

print("학습 가능한 파라미터:")
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=config.FINETUNED_MODEL_PATH,
    per_device_train_batch_size=config.BATCH_SIZE,
    num_train_epochs=config.EPOCHS,
    learning_rate=config.LEARNING_RATE,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="adamw_torch",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    fp16=True,
    bf16=False,
    torch_compile=False,
    dataloader_pin_memory=False,
    max_grad_norm=1.0,
    warmup_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
)

print("훈련(Fine-tuning)을 시작합니다")
print("=" * 50)
try:
    trainer.train()
    print("\n훈련 완료!")
except Exception as e:
    print(f"\n훈련 중 오류 발생: {e}")
    print("배치 사이즈를 줄이거나 gradient_accumulation_steps를 늘려보세요.")
    raise

final_save_path = os.path.join(config.FINETUNED_MODEL_PATH, "final_checkpoint")
trainer.save_model(final_save_path)
print(f"최종 어댑터가 {final_save_path}에 저장되었습니다.")