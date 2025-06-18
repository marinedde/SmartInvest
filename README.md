# SmartInvest

SmartInvest est un projet de data science visant à **analyser, visualiser et prédire les prix immobiliers à Paris** sur plusieurs années.  
Il a été développé dans le cadre du bootcamp **Jedha – Data Fullstack**.

## 🎯 Objectifs

- Étudier l'évolution des prix au m² à Paris de 2018 à 2024
- Détecter les facteurs influençant le marché immobilier (surface, arrondissement, année, etc.)
- Proposer une **application Streamlit** interactive pour explorer les données et faire des prédictions
- Utiliser **MLflow** pour le suivi des expérimentations modèles

## 📁 Structure du projet

SmartInvest/
├── README.md # Présentation du projet
├── data/ # Données brutes et nettoyées
├── notebooks/ # Notebooks d'analyse et ML
├── streamlit_app/ # Code de l'application Streamlit
├── mlflow/ # Dossier de tracking MLflow
├── models/ # Modèles sauvegardés (.pkl)
├── requirements.txt # Librairies nécessaires
└── .gitignore # Fichiers à exclure du versionning


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
