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