# imports
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from datasets import Dataset
import pandas as pd
from sklearn.metrics import classification_report

# csv laden
df = pd.read_csv('./train.csv')  # Stelle sicher, dass deine Datei den richtigen Pfad hat
df = df.sample(frac=0.1, random_state=42)  # Reduziere die Datenmenge (optional)

# one-hot encoding der labels
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(df['toxic'].values)

# DistilBERT-Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# tokenisiere des text (texte zu token-IDs )
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

# split in trainings- und validierungsdatensatz (80%/20%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['comment_text'].tolist(), labels, test_size=0.2, random_state=42, stratify=labels
)

# datasets
train_dataset = Dataset.from_dict({'text': train_texts, 'label': torch.tensor(train_labels)})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': torch.tensor(val_labels)})

# tokenizer auf dataset 
train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# DistilBERT mit zwei ausgabeklassen
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# args definieren (config)
training_args = TrainingArguments(
    output_dir='./results',          # Verzeichnis für das Ergebnis
    num_train_epochs=3,              # Anzahl der Epochen
    per_device_train_batch_size=8,   # Batch-Größe pro Gerät
    logging_dir='./logs',            # Verzeichnis für Logs
    evaluation_strategy="epoch",     # Evaluierung nach jeder Epoche
    save_strategy="epoch",           # Speichern nach jeder Epoche
    report_to=None,                  # Verhindere Verwendung von WANDB
    disable_tqdm=False
)

# trainer
trainer = Trainer(
    model=model,                     # Das Modell
    args=training_args,              # Trainingsargumente
    train_dataset=train_dataset,     # Trainingsdatensatz
    eval_dataset=val_dataset,        # Validierungsdatensatz
    tokenizer=tokenizer,             # Verwende den Tokenizer
)

# training starten
trainer.train()

# evaluierung und metriken nach fine-tuning
predictions, true_labels, _ = trainer.predict(val_dataset)
predictions_labels = predictions.argmax(axis=1)  # Wahrscheinlichkeiten in Labels umwandeln

# generierung des reports
report = classification_report(true_labels, predictions_labels, target_names=["Non-Toxic", "Toxic"], zero_division=1)
print(report)

# gpu nutzen wenn möglich
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ansi farben für ausgabe
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

# tatsächliche klassifikizierung
def classify_message(message: str):
    inputs = tokenizer(message, padding='max_length', truncation=True, max_length=128, return_tensors="pt").to(device)
  
    with torch.no_grad():
        logits = model(**inputs).logits

    #(0 = Non-Toxic, 1 = Toxic)
    predicted_class = torch.argmax(logits, dim=-1).item()
  
    if predicted_class == 1:
        print(f"{RED}Message Blocked{RESET}")  # toxic -> red
    else:
        print(f"{GREEN}{message}{RESET}")  # Non-toxic -> green

#
##
### "game loop"
##
#
print("\n >Toxic Chat Filter (Type 'exit' to quit)\n")

while True:
    user_input = input("Enter message: ")
    
    if user_input.lower() == "exit":
        print("Exiting...")
        break
    
    classify_message(user_input)
