# 🤝 README Collaboration – Projet SmartInvest

## 🎯 Problématique

> **Comment savoir sur quel logement investir à Paris ?**

Ce projet vise à aider un utilisateur à :
- Estimer la valeur d’un logement
- Comparer cette valeur avec les prix locaux
- Visualiser l’évolution des prix dans le temps et dans l’espace
- Identifier les biens potentiellement **sous-évalués**

---

## 📚 Répartition des rôles (3 membres)

### 🔷 Bloc A – Modélisation & Tracking ML (MLflow)
**Responsable : Marine**

- Nettoyage des données DVF
- Création de modèles (Random Forest, XGBoost, etc.)
- Comparaison des scores (R², MAE)
- Utilisation de **MLflow** pour suivre les expériences
- Sauvegarde du modèle final (`.pkl`)

### 🔷 Bloc B – Feature Engineering & Enrichissement
**Responsable : Membre 2**

- Ajout de features :
  - `annee`, `arrondissement`
  - `prix_m2_moyen_100m` (via KDTree)
  - `distance_metro`, `distance_ecole`, `nb_commerces_300m`
- Géocodage avec `geopy` si besoin
- Enrichissement spatial avec données OpenData ou scraping

### 🔷 Bloc C – Application & Visualisation (Streamlit)
**Responsable : Membre 3**

- Création d’une interface utilisateur avec Streamlit
- Formulaire : surface, code postal, pièces, année…
- Résultat :
  - Estimation du prix/m²
  - Comparaison avec moyenne quartier + bien similaires
  - Graphique évolution prix sur 5 ans
  - Résumé “Bon plan ou non ?”

---

## 📁 Arborescence du projet

SmartInvest/
├── notebooks/
│ ├── 01_cleaning_model_training.ipynb
│ └── 02_feature_engineering.ipynb
├── streamlit_app/
│ └── app.py
├── data/
│ ├── dvf_clean.csv
│ └── enriched_features.csv
├── models/
│ └── best_model.pkl
├── mlflow/
├── README.md
├── README_collaboration.md
├── requirements.txt
├── .gitignore


---

## 🔀 Branches Git recommandées

| Branche | Rôle |
|---------|------|
| `main` | fusion finale commune |
| `ml-dev` | modélisation & MLflow |
| `features-dev` | enrichissement & features |
| `app-dev` | application Streamlit |

---

## ✅ À faire à chaque modification

1. Travailler dans ta branche (ex. `features-dev`)
2. Ajouter ton fichier :
   ```bash
   git add mon_fichier.py
3. Commit :
git commit -m "Ajout feature distance école"
4. Push
git push origin ma-branche
5. git push origin ma-branche


📌 Technologies utilisées
Pandas, NumPy, Scikit-learn

Streamlit

MLflow

KDTree (recherche spatiale)

APIs ou scraping (géocodage, écoles, commerces)


---

## 💡 Pourquoi intégrer un LLM ?

Un LLM (modèle de langage comme ChatGPT, Mistral, Claude…) pourrait :

1. **Expliquer à l'utilisateur pourquoi un bien est (ou non) un bon investissement**
2. **Traduire des indicateurs techniques** (ex. “prix/m² 15% sous le marché”) en langage naturel
3. **Répondre à des questions posées par l’utilisateur** du type :

   * “Ce bien est-il rentable ?”
   * “Quel arrondissement a le plus progressé en 5 ans ?”
   * “Pourquoi ce logement est une opportunité ?”

---

## ✅ Ce qu'on peut faire **sans GPU et gratuitement**

### 1. **Utiliser un petit modèle local (open source)**

* **Mistral 7B**, **Phi-2**, ou **TinyLlama** peuvent tourner **en local**, même sans GPU si on utilise `llama.cpp`, `ggml`, ou **Google Colab**.
* **LangChain** ou **Transformers + PEFT** permettent de charger un modèle réduit **et le spécialiser**.

Tu peux créer un **chatbot local**, par exemple :

```python
from transformers import pipeline

chatbot = pipeline("text-generation", model="tiiuae/falcon-rw-1b")
prompt = "Est-ce que ce bien est une bonne opportunité si son prix au m² est 15% en dessous de la moyenne locale ?"

reponse = chatbot(prompt, max_new_tokens=100)
print(reponse[0]['generated_text'])
```

---

### 2. **Utiliser un LLM via API gratuite (ou quasi gratuite)**

* **OpenAI** (avec GPT-3.5-turbo) : coût très bas (0.0015 \$/1k tokens)
* **Groq** (héberge gratuitement Mixtral, LLaMA 3, Gemma)
* **HuggingFace Inference API** (certains modèles sont gratuits via le hub)

Tu pourrais intégrer dans Streamlit une fonction comme :

```python
import openai

openai.api_key = "YOUR_API_KEY"

def analyse_llm(contexte):
    prompt = f"Voici les infos d’un bien : {contexte}. Est-ce un bon investissement ? Justifie."

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
```

---

## 📦 Exemple concret dans SmartInvest :

Dans ton app Streamlit :

```python
# Résumé structuré
context = f"""
Prix : {prix_m2} €/m²
Prix moyen du quartier : {prix_m2_local} €/m²
Distance métro : {distance_metro} m
Nb commerces : {nb_commerces_300m}
"""
commentaire = analyse_llm(context)
st.markdown("### 💬 Analyse LLM :")
st.write(commentaire)
```

---

## 🎯 En résumé

| Solution                                 | Avantages                    | Coût                           |
| ---------------------------------------- | ---------------------------- | ------------------------------ |
| ✅ API GPT-3.5                            | Ultra simple, réponse rapide | 💸 faible (0.002 \$/1k tokens) |
| ✅ Open source local (Mistral 7B, Phi-2…) | Gratuit, sans cloud          | ⚠️ lent sans GPU               |
| ✅ HuggingFace + LangChain                | Plus personnalisable         | 💰 gratuit avec limite         |
| ❌ GPT-4 + agents + long context          | Trop cher, trop complexe     | ❌ pas pour ce projet étudiant  |

---


---

## 🧰 2. Comment l’ajouter à Git

### ▶️ Depuis ton terminal VS Code :

```bash
cd SmartInvest
touch README_collaboration.md
# (ou crée-le depuis l'explorateur VS Code)


🔁 Git Pull & Merge — Que faire si un fichier MERGE_MSG apparaît ?
Lorsque vous effectuez un git pull, Git fusionne les changements distants dans votre branche locale. Si Git détecte des différences, il vous demande de valider un message de merge.

Étapes à suivre :
Un fichier nommé MERGE_MSG s'ouvre automatiquement dans l'éditeur (souvent VS Code).

Laissez le message proposé par défaut (ex : Merge branch 'master' of https://github.com/...).

Enregistrez le fichier (Ctrl + S ou Cmd + S).

Fermez le fichier ou l’onglet.

Git finalise le merge automatiquement.

Vous pouvez ensuite pousser vos modifications :

bash
git push origin master



