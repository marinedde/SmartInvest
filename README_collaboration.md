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

## 🧰 2. Comment l’ajouter à Git

### ▶️ Depuis ton terminal VS Code :

```bash
cd SmartInvest
touch README_collaboration.md
# (ou crée-le depuis l'explorateur VS Code)



