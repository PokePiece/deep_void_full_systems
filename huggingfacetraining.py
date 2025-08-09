from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["input"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    logging_dir="./logs",
    num_train_epochs=3,  
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

trainer.train()









#model.save_pretrained("./fine_tuned_model")
#tokenizer.save_pretrained("./fine_tuned_model")


#fine_tuned_model = AutoModelForCausalLM.from_pretrained("./fine_tuned_model")
#fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
