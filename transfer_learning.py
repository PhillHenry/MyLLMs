from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from Tokenize import INPUT_TOKENIZED_DATASET
from config import model_name

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenized_dataset = AutoTokenizer.from_pretrained(INPUT_TOKENIZED_DATASET)

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
