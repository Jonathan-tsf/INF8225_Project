import os
import pandas as pd
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# Configuration de l'API
api_key = os.environ["MISTRAL_API_KEY"]

model = "mistral-large-latest"
client = MistralClient(api_key=api_key)

# Charger les données
if MMLU:
    df = pd.read_csv('MMLU-mini.csv')
else:
    df = pd.read_csv('GPQA-mini.csv')

# Vérification des colonnes nécessaires
expected_columns = {'prompt', 'A', 'B', 'C', 'D', 'answer'}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"Le DataFrame doit contenir les colonnes {expected_columns}.")

# Fonction générique pour envoyer une requête chat à Mistral
def mistral_chat_request(prompt):
    messages = [ChatMessage(role="user", content=prompt)]
    response = client.chat(model=model, messages=messages)
    return response.choices[0].message.content

# Traduire le texte en Lithuanien
def translate_to_lithuanian(text):
    return mistral_chat_request(f"Translate the following text to Lithuanian: {text}")

# Traduire le texte en Français (correction de l'orthographe de "French")
def translate_to_french(text):
    return mistral_chat_request(f"Translate the following text to French: {text}")

# Introduire des fautes d'orthographe
def introduce_spelling_errors(text):
    return mistral_chat_request(f"Introduce 20 spelling errors into this French text: {text}")

# Convertir le texte en langage soutenu
def convert_to_formal_language(text):
    return mistral_chat_request(f"Convert this French text into a more formal language: {text}")

# Fonction pour interroger le modèle avec le prompt transformé
def ask_gpt(question, options):
    prompt = f"{question}\nA: {options['A']}\nB: {options['B']}\nC: {options['C']}\nD: {options['D']}"
    return mistral_chat_request(prompt)

# Appliquer les transformations sur les questions et interroger l'API
results = []
correct = 0
total = len(df)
for index, row in df.iterrows():
    question = row['prompt']
    
    # Choisir la méthode souhaitée
    question = translate_to_lithuanian(question)  
  
    options = {'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D']}
    predicted_answer_full = ask_gpt(question, options)
    predicted_answer = predicted_answer_full.split(':')[0].strip()
    correct_answer = row['answer'].strip()
    results.append({
        'Question': question,
        'Option A': options['A'],
        'Option B': options['B'],
        'Option C': options['C'],
        'Option D': options['D'],
        'Predicted Answer': predicted_answer_full,
        'Correct Answer': correct_answer
    })
    if predicted_answer.upper() == correct_answer.upper():
        correct += 1

# Convertir la liste en DataFrame puis l’enregistrer en CSV
results_df = pd.DataFrame(results)
results_df.to_csv('results.csv', index=False)

# Affichage du score total
print(f"Score final : {correct}/{total} ({correct/total*100:.2f}%)")
