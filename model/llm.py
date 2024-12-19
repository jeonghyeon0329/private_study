from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch

# 1. 데이터 준비 (예시 데이터)
data = [
    "Hello, how are you?",
    "I am good, thank you!",
    "What is your name?",
    "I am a language model.",
    "How can I help you today?",
    "I am here to answer your questions."
]

# 데이터를 파일로 저장
with open('sample_data.txt', 'w') as f:
    for line in data:
        f.write(line + '\n')

# 2. 데이터셋 로딩
dataset = load_dataset('text', data_files={'train': 'sample_data.txt'}, split='train')

# 3. 토크나이저 로딩 (GPT-2용 토크나이저)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# pad_token 설정 (GPT-2는 기본적으로 pad_token이 없으므로 eos_token을 pad_token으로 설정)
tokenizer.pad_token = tokenizer.eos_token

# 4. 텍스트 데이터를 토큰화
def tokenize_function(examples):
    # 토큰화된 입력을 반환하고, labels은 input_ids와 동일하게 설정
    output = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=32)
    output["labels"] = output["input_ids"]  # labels는 input_ids와 동일하게 설정
    return output

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 5. GPT-2 모델 로딩
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 6. 훈련 인자 설정
training_args = TrainingArguments(
    output_dir='./results',          # 모델 체크포인트 저장 경로
    num_train_epochs=3,              # 훈련 에폭 수
    per_device_train_batch_size=2,   # 배치 크기
    save_steps=10_000,               # 체크포인트 저장 간격
    save_total_limit=2,              # 저장할 체크포인트의 개수 제한
)

# 7. Trainer 인스턴스 생성
trainer = Trainer(
    model=model,                         # 학습할 모델
    args=training_args,                  # 훈련 인자
    train_dataset=tokenized_datasets,    # 훈련 데이터셋
)

# 8. 모델 훈련
trainer.train()

# 9. 텍스트 생성
input_text = "Hello, how are"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# 모델에 텍스트를 입력하고 응답 생성
with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"Generated text: {generated_text}")
