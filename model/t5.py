from transformers import T5Tokenizer, T5ForConditionalGeneration

# T5 모델과 토크나이저 로드
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 대화 상태 추적 변수
conversation_history = ""

# 대화 함수 정의
def chat_with_ai(input_text):
    global conversation_history
    
    # 대화 이력을 포함하여 모델에 전달할 텍스트 준비 (T5는 조건부 생성이므로 이를 "대화:" 형식으로 명시)
    conversation_history += f"User: {input_text}\nAI:"
    
    # 대화 텍스트를 토크나이즈
    input_ids = tokenizer.encode(conversation_history, return_tensors="pt")
    
    # 모델로부터 응답 생성
    outputs = model.generate(input_ids, max_length=512, num_beams=5, early_stopping=True)
    
    # 모델의 응답 디코딩
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # AI의 응답만 추출
    ai_response = response.split("AI:")[-1].strip()
    
    # 새로운 대화 이력 갱신
    conversation_history += f" {ai_response}\n"
    
    return ai_response

# 대화 시작
print("대화형 AI 시작! 'exit'을 입력하면 대화를 종료합니다.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("대화 종료")
        break
    bot_response = chat_with_ai(user_input)
    print("AI: " + bot_response)
