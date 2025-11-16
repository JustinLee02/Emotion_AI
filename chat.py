import config
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os


# --- 1. 훈련 때와 '완전히 동일한' 4비트 설정 ---
# (bfloat16 대신 float16을 사용했던 것이 핵심)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# --- 2. 베이스 모델 로드 (Llama 3.1) ---
print(f"베이스 모델 로드 중: {config.BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    config.BASE_MODEL_ID,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# --- 3. [⭐️핵심] 'Growit 어댑터' 덮어씌우기 ---
# 'final_checkpoint' 경로를 config에서 가져옴
adapter_path = os.path.join(config.FINETUNED_MODEL_PATH, "final_checkpoint")

print(f"'{adapter_path}'에서 Growit 어댑터를 로드합니다...")

# PeftModel을 사용해 베이스 모델 위에 훈련된 어댑터를 덮어씌움
model = PeftModel.from_pretrained(base_model, adapter_path)

# 4비트 모델 + Peft 어댑터를 훈련이 아닌 '추론(evaluation)' 모드로 설정
model = model.eval()

print("--- 🤖 Growit AI (Finetuned) 준비 완료 ---")
print("('exit' 입력 시 종료)")

# --- 4. 채팅 루프 ---
history = [] # 간단한 대화 기록
system_message = {"role": "system", "content": "당신은 사용자의 일기에 공감하며 대화하는 친구 'Growit'입니다."}

while True:
    try:
        prompt = input("User: ")
        if prompt.lower() == "exit":
            break

        # 'messages' 형식 구성 (시스템 메시지 + 이전 대화 + 현재 입력)
        messages = [system_message] + history + [{"role": "user", "content": prompt}]
        
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)
        
        # --- 5. 모델 답변 생성 (이 부분이 10분 걸림) ---
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id # pad_token 설정
        )
        
        response_ids = outputs[0][input_ids.shape[-1]:]
        result_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"\nGrowit AI: {result_text}")
        
        # 대화 기록에 추가
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": result_text})

    except KeyboardInterrupt:
        print("\n채팅을 종료합니다.")
        break