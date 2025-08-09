import os
    

from llama_cpp import Llama

model_path = "../llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

print(f"Calculated model path: {model_path}")

llm = Llama(model_path=model_path, n_ctx=1000)

print("Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() == "exit":
        break

    response = llm(prompt=user_input, max_tokens=1000)
    print("AI:", response['choices'][0]['text'].strip())






'''
from llama_cpp import Llama

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "llama.cpp", "models", "mistral-7b-instruct-v0.1.Q4_K_M.gguf")

llm = Llama(model_path=model_path, n_ctx=4096, verbose=False)

def generate_response(prompt_text: str, max_tokens: int = 128):

    try:
        response = llm(prompt=prompt_text, max_tokens=max_tokens)
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Llama model error: {e}")
        return None
'''
    

