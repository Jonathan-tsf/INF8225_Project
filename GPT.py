OPENAI_API_KEY = "MOCK_KEY_FOR_GITHUB"

from openai import OpenAI
import pandas as pd

# Initialisation du client API OpenAI
client = OpenAI()

# Charger les données
if MMLU:
    df = pd.read_csv('MMLU-mini.csv')
else:
    df = pd.read_csv('GPQA-mini.csv')

# Vérification des colonnes nécessaires
expected_columns = {'prompt', 'A', 'B', 'C', 'D', 'answer'}
if not expected_columns.issubset(df.columns):
    raise ValueError(f"Le DataFrame doit contenir les colonnes {expected_columns}.")

# Traduire le texte en Lithuanien
def translate_to_lithuanian(text):
    translation = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[{"role": "system", "content": "Translate the following text to Lithuanian."}, 
                  {"role": "user", "content": text}]
    )
    return translation.choices[0].message.content


# Traduire le texte en Lithuanien
def translate_to_frensh(text):
    translation = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[{"role": "system", "content": "Translate the following text to Frensh."}, 
                  {"role": "user", "content": text}]
    )
    return translation.choices[0].message.content

# Introduire des fautes d'orthographe
def introduce_spelling_errors(text):
    errored_text = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0.9,  # Plus de créativité pour introduire des erreurs
        messages=[{"role": "system", "content": "Introduce 20 spelling errors into this French text."}, 
                  {"role": "user", "content": text}]
    )
    return errored_text.choices[0].message.content

# Convertir le texte en langage soutenu
def convert_to_formal_language(text):
    formal_text = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[{"role": "system", "content": "Convert this French text into a more formal language."}, 
                  {"role": "user", "content": text}]
    )
    return formal_text.choices[0].message.content

# Fonction pour interroger GPT-4 Turbo avec le prompt transformé
def ask_gpt(question, options):
    completion = client.chat.completions.create(
        model="gpt-4-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are an assistant trained to select the correct answer from provided options. Respond by starting with the letter of the correct answer followed by a colon and a space."},
            {"role": "user", "content": f"{question}\nA: {options['A']}\nB: {options['B']}\nC: {options['C']}\nD: {options['D']}"}
        ]
    )
    return completion.choices[0].message.content

# Appliquer les transformations sur les questions et interroger l'API
results = []
correct = 0
total = len(df)
for index, row in df.iterrows():
    question = row['prompt']
    
    # Choisir la méthode shouaiter
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
