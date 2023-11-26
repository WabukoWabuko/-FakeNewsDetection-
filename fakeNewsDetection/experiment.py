# %% Import packages and setup training parameters and seed (this is an interactive python file for VS Code)
import os.path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from transformers import (BertTokenizerFast, Trainer, TrainingArguments)

from utils import *

RANDOM_STATE = 49

# 0 = don't use summary, 1 = abstractive, 2 = extractive
IS_USE_SUMMARY_TRAIN = 0
IS_USE_SUMMARY_TEST = 0

# %% Generate extractive & abstractive summaries for both files if necessary
training_file = "training.csv"
testing_file = "testing.csv"
if not os.path.isfile(training_file):
    df = pd.read_csv("task_3a_sample_data.csv", sep="\t", header=0)
    df_2 = pd.read_csv("Task3a_training.csv", header=0)
    df = df.append(df_2)
    generate_extractive_summaries(df, training_file)
    df = pd.read_csv(training_file)
    generate_abstractive_summaries(df, training_file)

if not os.path.isfile(testing_file):
    df = pd.read_csv("Task3a_testing.csv", header=0)
    generate_extractive_summaries(df, testing_file)
    df = pd.read_csv(testing_file)
    generate_abstractive_summaries(df, testing_file)


# %% Convert rating to numeric value
df = pd.read_csv("training.csv")
df['label'] = df['our rating'].apply(convert_to_int)
# %% Split dataset into train and val set while keeping distribution & load test dataset
df_train, df_val = train_test_split(
    df, test_size=0.2, stratify=df['our rating'], random_state=RANDOM_STATE)

df_test = pd.read_csv("testing.csv")
# %% Oversample minority classes to ensure a better training
ros = RandomOverSampler()
x_resampled, y_resampled = ros.fit_resample(
    df_train.iloc[:, 0:-1], df_train["label"])
data_oversampled = pd.concat(
    [pd.DataFrame(x_resampled), pd.DataFrame(y_resampled)], axis=1)
df_train = data_oversampled

# %% Tokenize all texts and instantiate the dataset and training arguments
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
train_encodings, train_labels = get_encodings(
    df_train, tokenizer, IS_USE_SUMMARY_TRAIN)
val_encodings, val_labels = get_encodings(
    df_val, tokenizer, IS_USE_SUMMARY_TEST)

test_encodings = get_encodings_test(
    df_test, tokenizer, IS_USE_SUMMARY_TEST)

train_dataset = CheckThatLabDataset(train_encodings, train_labels)
val_dataset = CheckThatLabDataset(val_encodings, val_labels)
test_dataset = CheckThatLabDatasetTest(test_encodings)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    evaluation_strategy="steps",
    per_device_train_batch_size=8,   # batch size per device during training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,
    load_best_model_at_end=True,
    seed=RANDOM_STATE,
)
# %% Generate the Trainer object to train the model with
trainer = Trainer(
    # the instantiated 🤗 Transformers model to be trained
    model_init=init_full_text_model,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    data_collator=None,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# %% Train model
trainer.train()
# %% Evaluate model
evaluation_results = trainer.evaluate()
print(evaluation_results)
# %% Generate predictions
pred = trainer.predict(test_dataset)
preds = pred.predictions.argmax(-1)
df_test['label'] = preds
df_test['our rating'] = df_test['label'].apply(convert_to_rating)
columns = ["public_id", "our rating"]
# %% Save predictions to csv in desired format
if(IS_USE_SUMMARY_TRAIN == 0):
    df_test.to_csv("predictions.csv", columns=columns, index=False)
elif(IS_USE_SUMMARY_TRAIN == 1):
    df_test.to_csv("predictions_abstractive.csv", columns=columns, index=False)
else:
    df_test.to_csv("predictions_extractive.csv", columns=columns, index=False)