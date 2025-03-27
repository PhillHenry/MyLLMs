from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo_name = "zijiechen156/DeepSeek-R1-Medical-CoT"

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = AutoModelForCausalLM.from_pretrained(repo_name)

model.eval()

prompt = "What are the early symptoms of diabetes?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Model Response:", response)