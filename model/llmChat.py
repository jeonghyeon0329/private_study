from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 1. 데이터셋 로딩 (대화형 데이터셋 DailyDialog 사용)
dataset = load_dataset('daily_dialog', trust_remote_code=True)

# 2. 데이터셋의 필드 이름 확인 (디버깅을 위해)
print("Train dataset columns:", dataset['train'].column_names)

# 3. GPT-2 모델과 토크나이저 로딩
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# pad_token 설정 (GPT-2는 기본적으로 pad_token이 없으므로 eos_token을 pad_token으로 설정)
tokenizer.pad_token = tokenizer.eos_token

# 4. 데이터 전처리 함수 (토큰화 및 레이블 생성)
def tokenize_function(examples):
    # 데이터셋 구조에 맞게 대화 내용 필드 수정
    dialogues = examples['dialog']  # 이 부분을 실제 대화 내용이 있는 필드명으로 수정
    output = tokenizer(dialogues, padding="max_length", truncation=True, max_length=32)
    output["labels"] = output["input_ids"]  # labels는 input_ids와 동일하게 설정
    return output

# 5. 데이터셋에 토큰화 함수 적용
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 6. 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',          # 모델 체크포인트 저장 경로
    num_train_epochs=3,              # 훈련 에폭 수
    per_device_train_batch_size=2,   # 배치 크기
    save_steps=10_000,               # 체크포인트 저장 간격
    save_total_limit=2,              # 저장할 체크포인트의 개수 제한
    logging_steps=100,               # 로깅 빈도 설정
    evaluation_strategy="epoch"      # 훈련 중 평가 전략 설정
)

# 7. GPT-2 모델 로딩
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 8. Trainer 인스턴스 생성
trainer = Trainer(
    model=model,                         # 학습할 모델
    args=training_args,                  # 훈련 인자
    train_dataset=tokenized_datasets['train'],    # 훈련 데이터셋
    eval_dataset=tokenized_datasets['validation'],  # 검증 데이터셋
)

# 9. 모델 훈련
trainer.train()

# 10. 대화형 응답 생성 함수
def chat_with_model(input_text, previous_conversation=""):
    """
    대화형 AI 모델과의 대화를 처리하는 함수입니다.
    이전 대화 내용도 포함하여, 모델에게 맥락을 제공하고 응답을 생성합니다.
    """
    # 이전 대화 내용과 새로운 사용자 입력을 합칩니다.
    conversation = previous_conversation + f"User: {input_text}\nBot:"

    # 텍스트를 토큰화합니다.
    input_ids = tokenizer.encode(conversation, return_tensors='pt')

    # 모델에 텍스트를 입력하고 응답 생성
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    # 생성된 텍스트를 디코딩합니다.
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Bot의 응답만 추출 (대화 중 "Bot:" 뒤의 텍스트만 반환)
    bot_response = generated_text.split("Bot:")[-1].strip()

    # 새롭게 생성된 응답을 이전 대화에 추가하여 반환
    return bot_response, generated_text

# 11. 대화형 응답 테스트
previous_conversation = ""
while True:
    user_input = input("You: ")  # 사용자 입력을 받습니다.
    
    # "exit" 입력 시 대화 종료
    if user_input.lower() == "exit":
        print("Ending the conversation.")
        break

    # 모델과 대화하고 응답을 받음
    bot_response, full_conversation = chat_with_model(user_input, previous_conversation)
    
    # 봇의 응답 출력
    print(f"Bot: {bot_response}")
    
    # 대화 내용 업데이트 (대화 내용에 이전 대화와 봇의 응답을 포함)
    previous_conversation = full_conversation
