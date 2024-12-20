from transformers import pipeline

# DialoGPT 모델 로드
chatbot = pipeline('text-generation', model='microsoft/DialoGPT-medium')

# 대화 시작
conversation_history = "Hello, how are you?"

# 첫 번째 응답 생성
response = chatbot(conversation_history, max_length=100, truncation=True, num_return_sequences=1)
bot_reply = response[0]['generated_text']

# 봇의 응답 출력
print(f"Bot: {bot_reply}")

# 대화를 이어가기
while True:
    # 사용자 입력 받기
    user_input = input("You: ")
    
    # 사용자가 'exit'을 입력하면 대화 종료
    if user_input.lower() == 'exit':
        print("Bot: Goodbye!")
        break

    # 사용자 입력만 대화 히스토리에 추가 (봇의 이전 응답은 다시 추가하지 않음)
    conversation_history += f" {user_input}"

    # 모델 응답 생성
    response = chatbot(conversation_history, max_length=100, truncation=True, num_return_sequences=1)
    bot_reply = response[0]['generated_text']

    # 봇의 새로운 응답 출력
    print(f"Bot: {bot_reply}")

    # 대화 히스토리를 너무 길어지지 않게 관리 (예: 최대 1000자까지만 유지)
    if len(conversation_history) > 1000:
        conversation_history = conversation_history[-1000:]  # 최근 1000자를 남기고 자르기
