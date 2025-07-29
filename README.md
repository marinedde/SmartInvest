<<<<<<< HEAD
cat > README.md << 'EOF'
# 🏠 SmartInvest - Estimation Immobilière Paris

> Application d'estimation de prix immobilier pour Paris utilisant l'Intelligence Artificielle

## 🎯 Description

SmartInvest utilise un modèle XGBoost entraîné sur les données DVF pour estimer le prix au m² des biens immobiliers parisiens en temps réel.

## 🚀 Démarrage rapide

### Installation
```bash
git clone https://github.com/marinedde/smartinvest.git
cd smartinvest
pip install -r requirements.txt

Lancement
bash# Terminal 1 - API
python main.py

# Terminal 2 - Interface
streamlit run app.py
Accès

Interface utilisateur : http://localhost:8501
API REST : http://localhost:8001
Documentation API : http://localhost:8001/docs

📊 Performance du modèle
MétriqueValeurR² Score34.2%MAE±1,595 €/m²RMSE2,223 €/m²AlgorithmeXGBoost RegressorFeatures46 variables
✨ Fonctionnalités

🏠 Estimation instantanée du prix au m²
📊 Visualisations interactives par arrondissement
🎯 Marges d'erreur transparentes
🔧 Options avancées (balcon, parking, ascenseur)
📈 Comparaison avec moyennes de marché
🧪 Interface de test avec vraies annonces

🏗️ Architecture
├── main.py              # API FastAPI
├── app.py               # Interface Streamlit
├── models/model.pkl     # Modèle XGBoost entraîné
├── test_real_annonces.py # Tests de validation
└── requirements.txt     # Dépendances
🧪 Tests
Test avec des annonces réelles :
bashpython test_real_annonces.py
🛠️ Technologies

Backend : FastAPI, XGBoost, scikit-learn
Frontend : Streamlit, Plotly
Data : Pandas, NumPy
Déploiement : Uvicorn, Docker ready

👥 Équipe
Projet réalisé dans le cadre de la formation Data Science Fullstack chez JEDHA.
📄 Licence
MIT License - Projet éducatif

🏡 SmartInvest - L'estimation immobilière intelligente pour Paris EOF
=======
# SmartInvest

SmartInvest est un projet de data science visant à **analyser, visualiser et prédire les prix immobiliers à Paris** sur plusieurs années.  
Il a été développé dans le cadre du bootcamp **Jedha – Data Fullstack**.

## Utilisation et venv

Pour utiliser les scripts et notebook correctement un fichier requirements.txt est disponible
Il est fortement recommandé d'utiliser un environnement virtuel Python (venv) :
cmd : python -m venv venv

Puis installer les dépendance via : 
cmd : pip install -r requirements.txt


## 🎯 Objectifs

- Étudier l'évolution des prix au m² à Paris de 2018 à 2024
- Détecter les facteurs influençant le marché immobilier (surface, arrondissement, année, etc.)
- Proposer une **application Streamlit** interactive pour explorer les données et faire des prédictions
- Utiliser **MLflow** pour le suivi des expérimentations modèles

<!-- ## 📁 Structure du projet

SmartInvest/
├── README.md # Présentation du projet
├── data/ # Données brutes et nettoyées
├── notebooks/ # Notebooks d'analyse et ML
├── streamlit_app/ # Code de l'application Streamlit
├── mlflow/ # Dossier de tracking MLflow
├── models/ # Modèles sauvegardés (.pkl)
├── requirements.txt # Librairies nécessaires
└── .gitignore # Fichiers à exclure du versionning -->


## 🔧 Technologies utilisées

- Python (Pandas, Scikit-learn, Matplotlib, etc.)
- Streamlit
- MLflow
- Git / GitHub

## 👥 Collaborateurs

- [@tonpseudoGitHub](https://github.com/tonpseudoGitHub)
- [@pseudoCollaborateur](https://github.com/pseudoCollaborateur)

## 🚀 Lancement de l’application

```bash
cd streamlit_app
streamlit run app.py
>>>>>>> 6809313bb84ee918501f1263416fa810dae7617c
