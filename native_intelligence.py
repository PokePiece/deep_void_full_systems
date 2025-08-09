from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import io
import torch
from huggingface_hub import login

login(token='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')

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model_name = "EleutherAI/pythia-410m"

try:
    print("Loading model with bfloat16 to save memory...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", torch_dtype=torch.bfloat16)
    print("Model loaded successfully.")
except Exception as e:
    print(f"An error occurred during model loading: {e}")
    sys.exit(1)

def generate(prompt, profile):
    system_prompt = f"You're a {profile['tone']} assistant. Be {profile['style']}."
    full_prompt = system_prompt + "\nUser: " + prompt + "\nAI:"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.9,
        temperature=0.2, 
        repetition_penalty=1.2,
    )
    generated_tokens = outputs[0][inputs['input_ids'].size(1):]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if __name__ == "__main__":
    print("Native intelligence operating. Type 'quit' to exit.")
    while True:
        prompt = input("You: ")
        if prompt.lower() in ("quit", "exit"):
            print("Exiting... Goodbye!")
            break
        profile = {"tone": "realistic", "style": "technical"}
        response = generate(prompt, profile)
        print("AI:", response)