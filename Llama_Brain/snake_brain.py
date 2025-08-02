from llama_cpp import Llama

model_path = "../llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
llm = Llama(model_path=model_path)

print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break

    response = llm(prompt=user_input, max_tokens=128)
    print("AI:", response['choices'][0]['text'].strip())
