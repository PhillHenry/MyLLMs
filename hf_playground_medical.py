from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo_name = "zijiechen156/DeepSeek-R1-Medical-CoT"

tokenizer = AutoTokenizer.from_pretrained(repo_name)
model = AutoModelForCausalLM.from_pretrained(repo_name)

model.eval()

prompt = "Do the following ICD10 codes represent blood cancer: L01.0, L4.1, C00.0, D57.1?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=1024)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("Model Response:", response)