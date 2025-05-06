import transformers
import torch

model_id = "ContactDoctor/Bio-Medical-Llama-3-2-1B-CoT-012025"
# model_id = "medicalai/ClinicalBERT"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an expert trained on healthcare and biomedical domain!"},
    {"role": "user", "content": "Do the following ICD10 codes represent blood cancer: L01.0, L4.1, C00.0, D57.1"},
]

prompt = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    prompt,
    max_new_tokens=1024,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])