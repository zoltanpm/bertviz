import transformers as tf
import polars as pl
import datasets
import evaluate
import glob

# Get a list of all files that start with "synthetic_metaphors"
file_list = glob.glob('synthetic_metaphors*')

# Read and combine all files into one polars DataFrame
df_list = [pl.read_csv(file) for file in file_list]
combined_df = pl.concat(df_list)

ds = datasets.Dataset.from_polars(combined_df)
ds = ds.train_test_split(test_size=0.2)

from transformers import AutoTokenizer, AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# distilbert_num_parameters = model.num_parameters() / 1_000_000
# print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
# print(f"'>>> BERT number of parameters: 110M'")

def tokenize_function(examples):
    result = tokenizer(examples["texts"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = ds.map(
    tokenize_function, batched=True, remove_columns=["texts"])


from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# samples = [tokenized_datasets["train"][i] for i in range(2)]
# for sample in samples:
#     _ = sample.pop("word_ids")

# for chunk in data_collator(samples)["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")


from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="results/",
    #evaluation_strategy="epoch",
    eval_strategy="steps",
    num_train_epochs=40,
    save_steps=1000,
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)
#model = AutoModelForMaskedLM.from_pretrained('cp', local_files_only=True)

import math
eval_results = trainer.evaluate()
print(f">>> Base Model Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()
import math
eval_results = trainer.evaluate()
print(f">>> Tuned Model Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
model.save_pretrained('results/')
