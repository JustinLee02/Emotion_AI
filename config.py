# config.py 

# --- 모델 설정 ---
BASE_MODEL_ID = "moonshotai/Kimi-K2-Thinking"

DATA_PATH = "data/growit_dialogues.jsonl"

# --- QLoRA 학습 설정 ---
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
EPOCHS = 3

# --- 결과물 저장 경로 ---
FINETUNED_MODEL_PATH = "finetuned_models/growit_v1"