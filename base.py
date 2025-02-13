
#from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
#import torch
#from datasets import load_dataset
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device_name = torch.cuda.get_device_name(0)
#print(f"Arquitetura de dispositivo: '{device}' e nome do dispositivo: '{device_name}'" )
#
## 2. Load the pre-trained T5-small model and tokenizer
#model_name = "t5-small"
#model = T5ForConditionalGeneration.from_pretrained(model_name)
#model.to(device)


# 3. Load your dataset from the CSV file.
#    Make sure the CSV file has columns "problem" and "editorial".
#dataset = load_dataset("csv", data_files={"train": "data.csv"}, delimiter=",")["train"]
#
## 4. Preprocessing function: prepare the inputs and labels.
#def preprocess_function(examples):
#   # Create input text by prepending "Problem:" to each problem statement.
#   inputs = ["Problem: " + problem for problem in examples["problem"]]
#   targets = examples["editorial"]
#   
#   # Tokenize the inputs and targets.
#   model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
#   # Use the tokenizer in target mode for the labels.
#   with tokenizer.as_target_tokenizer():
#       labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
#   
#   model_inputs["labels"] = labels["input_ids"]
#   return model_inputs
#
## 5. Apply the preprocessing to the dataset.
#tokenized_dataset = dataset.map(preprocess_function, batched=True)
#tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
#
## 6. Define training arguments.
## Adjust parameters (epochs, batch size, learning rate) based on your resource constraints.
#training_args = TrainingArguments(
#   output_dir="./results",            # Where to store model checkpoints
#   num_train_epochs=3,                # Number of training epochs
#   per_device_train_batch_size=2,     # Adjust based on your GPU memory (RX 580 8GB might need a small batch size)
#   gradient_accumulation_steps=4,     # To effectively simulate a larger batch size
#   learning_rate=5e-5,
#   logging_steps=10,
#   save_strategy="epoch",
#   evaluation_strategy="no",          # Change if you have a validation set
#)
#
## 7. Initialize the Trainer.
#trainer = Trainer(
#   model=model,
#   args=training_args,
#   train_dataset=tokenized_dataset,
#)
#
## 8. Start the training process.
#trainer.train()
#
## (Optional) Save the final model.
#model.save_pretrained("./final_model")
#tokenizer.save_pretrained("./final_model")