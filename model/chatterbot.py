from transformers import pipeline

chatbot = pipeline('text-generation', model='microsoft/DialoGPT-medium')

# 대화 시작
conversation_history = "Hello, how are you?"

# 모델 응답 생성
response = chatbot(conversation_history, max_length=100, num_return_sequences=1)
print(f"Bot: {response[0]['generated_text']}")

# 사용자 입력에 반응하여 대화 이어가기
user_input = input("You: ")
conversation_history += f" {user_input}"
response = chatbot(conversation_history, max_length=100, num_return_sequences=1)
print(f"Bot: {response[0]['generated_text']}")
