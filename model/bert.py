from transformers import AutoTokenizer, AutoModelForCausalLM

# DialoGPT 모델과 토크나이저 불러오기
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# 대화 상태를 추적하기 위해 사용될 변수를 초기화
chat_history_ids = None

# 대화 함수 정의
def chat_with_ai(input_text):
    global chat_history_ids
    
    # 입력 텍스트를 토크나이즈
    new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")

    # 대화의 역사와 함께 모델에 입력을 전달
    bot_input_ids = new_user_input_ids if chat_history_ids is None else torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # 모델에서 답변 생성
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id, no_repeat_ngram_size=3, do_sample=True, top_k=100, top_p=0.7, temperature=0.8)

    # 모델의 응답을 디코드
    bot_output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    return bot_output

# 대화 예시
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    bot_response = chat_with_ai(user_input)
    print("AI: " + bot_response)
