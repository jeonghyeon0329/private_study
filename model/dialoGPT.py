from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# 대화 상태 추적 변수
chat_history_ids = None

# 대화 함수 정의
def chat_with_ai(input_text):
    global chat_history_ids
    
    # 사용자 입력을 토크나이즈 (EOS 토큰 추가)
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    # 대화 역사와 결합 (초기에는 없을 수 있음)
    bot_input_ids = new_user_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # 모델에서 응답 생성 (최대 길이 설정)
    chat_history_ids = model.generate(
        bot_input_ids, 
        max_length=1000, 
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        do_sample=True, 
        top_k=100, 
        top_p=0.7, 
        temperature=0.8
    )

    # 모델의 응답을 디코딩하여 반환
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return bot_output

# 대화 시작
print("대화형 AI 시작! 'exit'을 입력하면 대화를 종료합니다.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("대화 종료")
        break
    bot_response = chat_with_ai(user_input)
    print("AI: " + bot_response)
