from datasets import DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2TokenizerFast
from transformers import TrainingArguments, Trainer

from Tokenize import INPUT_TOKENIZED_DATASET
from config import MODEL_NAME

tokenized_dataset = DatasetDict.load_from_disk(INPUT_TOKENIZED_DATASET)
#tokenized_dataset = GPT2TokenizerFast.from_pretrained(INPUT_TOKENIZED_DATASET)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=5e-5,
    fp16=True,
    per_device_train_batch_size=4, # was 200
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch"
)

# Define the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"]
)

# Train the model
trainer.train()
