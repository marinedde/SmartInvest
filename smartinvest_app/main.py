# main.py - API SmartInvest CORRIGÉE (Sans valeur foncière) - VERSION FIXÉE 46 FEATURES
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from datetime import datetime
import os

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation FastAPI
app = FastAPI(
    title="🏠 SmartInvest API",
    description="API de prédiction de prix immobilier pour Paris (SANS valeur foncière)",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement modèle
model = None
model_path_used = None

paths_to_try = [
    "model.pkl",
    "./model.pkl", 
    "models/model.pkl",
    os.path.join(os.getcwd(), "model.pkl"),
]

for path in paths_to_try:
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            model_path_used = path
            logger.info(f"✅ Modèle chargé: {path}")
            break
    except Exception as e:
        logger.warning(f"⚠️ Erreur chargement {path}: {e}")

if model is None:
    logger.error("❌ AUCUN MODÈLE CHARGÉ")

# Schema Pydantic CORRIGÉ - SANS valeur foncière
class InputData(BaseModel):
    """Données d'entrée pour prédiction (SANS valeur foncière)"""
    
    # Features de base DVF (obligatoires)
    surface_reelle_bati: float = Field(
        ..., 
        gt=10.0, 
        le=500.0,
        description="Surface réelle bâtie en m²"
    )
    nombre_pieces_principales: int = Field(
        ..., 
        ge=1, 
        le=15,
        description="Nombre de pièces principales"
    )
    arrondissement: int = Field(
        ..., 
        ge=1, 
        le=20,
        description="Arrondissement parisien (1-20)"
    )
    
    # Coordonnées (importantes pour features enrichies)
    longitude: float = Field(
        default=2.347,
        ge=2.224,
        le=2.469,
        description="Longitude (coordonnée Est Paris)"
    )
    latitude: float = Field(
        default=48.857,
        ge=48.815,
        le=48.902,
        description="Latitude (coordonnée Nord Paris)"
    )
    
    # Features optionnelles avec valeurs par défaut
    nombre_lots: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Nombre de lots"
    )
    annee: int = Field(
        default=2024,
        ge=2020,
        le=2024,
        description="Année de référence"
    )
    annee_construction_dpe: int = Field(
        default=1970,
        ge=1800,
        le=datetime.now().year,
        description="Année de construction (base DPE)"
    )
    
    @validator('arrondissement')
    def validate_arrondissement(cls, v):
        if v not in range(1, 21):
            raise ValueError('Arrondissement doit être 1-20')
        return v

class PredictionResponse(BaseModel):
    """Réponse de prédiction CORRIGÉE"""
    prix_m2: float = Field(..., description="Prix prédit au m² (€)")
    valeur_totale_estimee: float = Field(..., description="Valeur totale pour la surface donnée")
    marge_erreur: float = Field(..., description="Marge d'erreur ±MAE (€/m²)")
    fourchette_min: float = Field(..., description="Prix minimum fourchette")
    fourchette_max: float = Field(..., description="Prix maximum fourchette") 
    confiance: float = Field(..., description="Niveau de confiance 0-1")
    details: Dict[str, Any] = Field(..., description="Détails de l'estimation")
    timestamp: str = Field(..., description="Horodatage")

# Fonctions utilitaires pour générer les 46 features
def encode_zone_manuelle(arrondissement: int) -> int:
    """Encode l'arrondissement en zones"""
    if arrondissement in [1, 2, 3, 4]:
        return 1  # Centre historique
    elif arrondissement in [5, 6, 7, 8]:
        return 2  # Centre-Ouest chic
    elif arrondissement in [9, 10, 11]:
        return 3  # Nord-Est populaire
    elif arrondissement in [12, 13, 14]:
        return 4  # Sud-Est
    elif arrondissement in [15, 16]:
        return 5  # Ouest résidentiel
    else:  # 17, 18, 19, 20
        return 6  # Nord périphérique

def etage_to_numeric(etage_str: str = "2") -> int:
    """Convertit l'étage en numérique (valeur par défaut)"""
    try:
        return int(etage_str) if etage_str.isdigit() else 2
    except:
        return 2

def encode_poi_dominant(distance_metro: float, nb_pois: int) -> int:
    """Encode le type de POI dominant"""
    if distance_metro < 0.3:
        return 0  # Transport dominant
    elif nb_pois > 15:
        return 1  # Commerce dominant
    elif nb_pois < 5:
        return 2  # Résidentiel
    else:
        return 3  # Mixte

def encode_zone_kmeans(arrondissement: int) -> int:
    """Encode en zones K-means simulées"""
    zones = {
        1: 0, 2: 0, 3: 1, 4: 0, 5: 1, 6: 0, 7: 0, 8: 0,
        9: 2, 10: 2, 11: 2, 12: 3, 13: 3, 14: 3, 15: 4,
        16: 4, 17: 5, 18: 5, 19: 5, 20: 2
    }
    return zones.get(arrondissement, 2)

def calculate_comprehensive_features(data: InputData) -> Dict[str, float]:
    """Calcule toutes les features de distance et POI nécessaires"""
    
    # Coordonnées approximatives du centre de chaque arrondissement
    arrond_centers = {
        1: (2.341, 48.860), 2: (2.342, 48.869), 3: (2.362, 48.863),
        4: (2.354, 48.854), 5: (2.351, 48.844), 6: (2.332, 48.851),
        7: (2.318, 48.856), 8: (2.313, 48.873), 9: (2.338, 48.876),
        10: (2.363, 48.873), 11: (2.379, 48.858), 12: (2.388, 48.840),
        13: (2.359, 48.829), 14: (2.327, 48.833), 15: (2.301, 48.842),
        16: (2.275, 48.848), 17: (2.308, 48.884), 18: (2.342, 48.892),
        19: (2.384, 48.884), 20: (2.397, 48.864)
    }
    
    center_lon, center_lat = arrond_centers.get(data.arrondissement, (data.longitude, data.latitude))
    base_distance = abs(data.longitude - center_lon) + abs(data.latitude - center_lat)
    
    # Calcul des distances
    distance_metro = max(0.1, min(2.5, base_distance * 50 + np.random.uniform(0.1, 0.8)))
    
    # Densité POI par arrondissement
    poi_density = {
        1: 25, 2: 20, 3: 22, 4: 24, 5: 18, 6: 26, 7: 20, 8: 23,
        9: 19, 10: 17, 11: 16, 12: 14, 13: 12, 14: 13, 15: 15,
        16: 18, 17: 16, 18: 14, 19: 11, 20: 13
    }
    
    base_poi_count = poi_density.get(data.arrondissement, 15)
    poi_multiplier = max(0.5, 2.0 - distance_metro)
    nb_pois = int(base_poi_count * poi_multiplier)
    
    return {
        "distance_metro_km": distance_metro,
        "distance_ecole_km": max(0.2, distance_metro * 0.8),
        "distance_college_km": distance_metro * 1.1,
        "distance_universite_km": distance_metro * 2.0,
        "distance_espace_vert_km": distance_metro * 0.9,
        "distance_commerce_km": distance_metro * 1.2,
        "distance_transport_km": distance_metro * 3.0,
        "distance_POI_min_km": min(distance_metro, 0.1),
        "distance_batiment_m": 25.0,
        "distance_TER_km": max(1.0, distance_metro * 5.0),
        "distance_datashop_km": max(0.5, distance_metro * 2.0),
        "nb_POIs_1km": nb_pois,
        "nb_POIs_<1km": nb_pois,
        "proche_POI_1km": 1 if nb_pois > 10 else 0,
        "POI_dominant": encode_poi_dominant(distance_metro, nb_pois)
    }

def prepare_model_features(data: InputData) -> pd.DataFrame:
    """
    Prépare EXACTEMENT les 46 features attendues par le modèle
    SANS utiliser la valeur foncière réelle
    """
    
    # Calculer toutes les features de distance et POI
    comprehensive_features = calculate_comprehensive_features(data)
    
    # Features de base
    age_batiment = datetime.now().year - data.annee_construction_dpe
    surface_par_piece = data.surface_reelle_bati / data.nombre_pieces_principales
    
    # Estimation du prix de référence basée sur l'arrondissement (SANS valeur foncière réelle)
    prix_ref_arr = {
        1: 15000, 2: 13000, 3: 12000, 4: 14000, 5: 11000, 6: 14500, 7: 13500, 8: 12500,
        9: 10500, 10: 10000, 11: 9500, 12: 9000, 13: 8500, 14: 9200, 15: 9800,
        16: 11500, 17: 9300, 18: 8800, 19: 8200, 20: 8500
    }
    prix_m2_reference = prix_ref_arr.get(data.arrondissement, 9000)
    valeur_estimee = prix_m2_reference * data.surface_reelle_bati
    
    # Construire le dictionnaire complet de 46 features
    features = {
        # Features de base (8)
        "surface_reelle_bati": data.surface_reelle_bati,
        "valeur_fonciere": valeur_estimee,  # Estimation basée sur arrondissement
        "annee": data.annee,
        "annee_construction_dpe": data.annee_construction_dpe,
        "nombre_pieces_principales": data.nombre_pieces_principales,
        "arrondissement": data.arrondissement,
        "distance_metro_km": comprehensive_features["distance_metro_km"],
        "nb_POIs_1km": comprehensive_features["nb_POIs_1km"],
        
        # Features géographiques (6)
        "latitude": data.latitude,
        "longitude": data.longitude,
        "x": 652000 + data.arrondissement * 1000,
        "y": 6862000 + data.arrondissement * 1000,
        "distance_centre_km": abs(data.arrondissement - 1) * 0.5,
        "zone_code": encode_zone_manuelle(data.arrondissement),
        
        # Features de distances (8)
        "distance_ecole_km": comprehensive_features["distance_ecole_km"],
        "distance_college_km": comprehensive_features["distance_college_km"],
        "distance_universite_km": comprehensive_features["distance_universite_km"],
        "distance_espace_vert_km": comprehensive_features["distance_espace_vert_km"],
        "distance_commerce_km": comprehensive_features["distance_commerce_km"],
        "distance_transport_km": comprehensive_features["distance_transport_km"],
        "distance_POI_min_km": comprehensive_features["distance_POI_min_km"],
        "distance_batiment_m": comprehensive_features["distance_batiment_m"],
        
        # Features calculées (8)
        "age_batiment": age_batiment,
        "surface_par_piece": surface_par_piece,
        "prix_m2_reference": prix_m2_reference,
        "densite_pieces": data.nombre_pieces_principales / data.surface_reelle_bati * 100,
        "etage_numerique": etage_to_numeric("2"),  # Valeur par défaut
        "bonus_equipements": 1,  # Valeur par défaut (pas de balcon/parking dans InputData)
        "surface_categorie": 1 if data.surface_reelle_bati < 50 else 2 if data.surface_reelle_bati < 80 else 3,
        "arrondissement_groupe": 1 if data.arrondissement <= 10 else 2,
        
        # Features de marché (6)
        "nb_lots": data.nombre_lots,
        "proche_POI": comprehensive_features["proche_POI_1km"],
        "poi_dominant_code": comprehensive_features["POI_dominant"],
        "zone_kmeans": encode_zone_kmeans(data.arrondissement),
        "market_segment": 1 if valeur_estimee < 400000 else 2 if valeur_estimee < 800000 else 3,
        "luxury_indicator": min(3, 1 + (1 if data.arrondissement <= 8 else 0)),
        
        # Features temporelles (4)
        "mois": datetime.now().month,
        "trimestre": (datetime.now().month - 1) // 3 + 1,
        "is_recent_construction": 1 if age_batiment < 10 else 0,
        "is_old_construction": 1 if age_batiment > 50 else 0,
        
        # Features d'interaction (6)
        "surface_x_pieces": data.surface_reelle_bati * data.nombre_pieces_principales,
        "prix_x_arrond": valeur_estimee * data.arrondissement / 1000000,
        "age_x_surface": age_batiment * data.surface_reelle_bati / 100,
        "metro_x_pois": comprehensive_features["distance_metro_km"] * comprehensive_features["nb_POIs_1km"],
        "pieces_x_arrond": data.nombre_pieces_principales * data.arrondissement,
        "surface_value_ratio": data.surface_reelle_bati / (valeur_estimee / 100000),
    }
    
    # Vérifier qu'on a exactement 46 features
    if len(features) != 46:
        logger.warning(f"⚠️ Nombre de features incorrect: {len(features)} au lieu de 46")
    
    # Créer DataFrame
    df = pd.DataFrame([features])
    
    # Si le modèle a des noms de features spécifiques, réorganiser
    if hasattr(model, 'feature_names_in_'):
        expected_features = model.feature_names_in_
        logger.info(f"📋 Modèle attend {len(expected_features)} features")
        
        # Créer un dictionnaire avec toutes les features attendues
        final_features = {}
        for feature_name in expected_features:
            if feature_name in features:
                final_features[feature_name] = features[feature_name]
            else:
                # Valeur par défaut pour features manquantes
                if 'distance' in feature_name.lower():
                    final_features[feature_name] = 1.0
                elif 'nombre' in feature_name.lower() or 'nb_' in feature_name.lower():
                    final_features[feature_name] = 1
                elif 'prix' in feature_name.lower() or 'valeur' in feature_name.lower():
                    final_features[feature_name] = prix_m2_reference
                else:
                    final_features[feature_name] = 0.0
        
        # Recréer le DataFrame avec les bonnes features
        df = pd.DataFrame([final_features])
        df = df[expected_features]  # Ordre exact du modèle
    
    # Conversion en float pour XGBoost
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def calculate_confidence(data: InputData, prediction: float) -> float:
    """Calcule confiance basée sur MAE=1596€/m² et MAPE=14.9%"""
    
    base_confidence = 0.75  # Basé sur MAPE de 14.9%
    
    # Ajustements
    confidence = base_confidence
    
    # Réduction pour cas atypiques
    surface_par_piece = data.surface_reelle_bati / data.nombre_pieces_principales
    if surface_par_piece < 15 or surface_par_piece > 50:
        confidence *= 0.85
    
    # Réduction pour prédictions extrêmes
    if prediction > 18000 or prediction < 4000:
        confidence *= 0.80
    
    # Bonus pour arrondissements avec plus de données
    if data.arrondissement in [11, 18, 19, 20, 15]:
        confidence *= 1.05
    elif data.arrondissement in [1, 4, 7, 8]:
        confidence *= 0.90
    
    return max(0.40, min(0.95, confidence))

# ENDPOINTS

@app.get("/")
async def root():
    return {
        "message": "🏠 SmartInvest API v3.0 - SANS valeur foncière - 46 features",
        "docs": "/docs",
        "health": "/health",
        "important": "Cette API prédit le prix au m² SANS utiliser la valeur foncière réelle"
    }

@app.get("/health")
async def health_check():
    expected_features = 46
    if model and hasattr(model, 'feature_names_in_'):
        expected_features = len(model.feature_names_in_)
    
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "version": "3.0.0",
        "features_expected": expected_features,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(data: InputData):
    """
    Prédiction du prix au m² SANS utiliser la valeur foncière réelle
    Utilise uniquement les caractéristiques du bien et sa localisation
    """
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle ML non disponible"
        )
    
    try:
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger.info(f"🔍 Prédiction [{request_id}] - {data.arrondissement}e arr., {data.surface_reelle_bati}m²")
        
        # Préparation features
        input_df = prepare_model_features(data)
        
        logger.info(f"📊 Features préparées: {input_df.shape[1]} colonnes")
        
        # PRÉDICTION
        prediction = model.predict(input_df)[0]
        
        # Validation réaliste
        prediction = max(3000, min(25000, prediction))
        
        # Calculs
        valeur_totale = prediction * data.surface_reelle_bati
        confidence = calculate_confidence(data, prediction)
        mae = 1596  # Votre MAE réelle
        
        fourchette_min = max(1000, prediction - mae)
        fourchette_max = min(30000, prediction + mae)
        
        # Détails
        details = {
            "request_id": request_id,
            "arrondissement": f"{data.arrondissement}e arrondissement",
            "surface_par_piece": round(data.surface_reelle_bati / data.nombre_pieces_principales, 1),
            "age_batiment": datetime.now().year - data.annee_construction_dpe,
            "categorie_prix": (
                "Premium" if prediction > 15000 
                else "Élevé" if prediction > 12000 
                else "Moyen" if prediction > 9000 
                else "Abordable"
            ),
            "methode": "XGBoost sans valeur foncière - 46 features",
            "features_utilisees": len(input_df.columns),
            "coordonnees": f"{data.latitude:.3f}, {data.longitude:.3f}"
        }
        
        logger.info(f"✅ Prédiction [{request_id}]: {prediction:.0f} €/m² (confiance: {confidence:.2f})")
        
        return PredictionResponse(
            prix_m2=round(prediction, 0),
            valeur_totale_estimee=round(valeur_totale, 0),
            marge_erreur=mae,
            fourchette_min=round(fourchette_min, 0),
            fourchette_max=round(fourchette_max, 0),
            confiance=round(confidence, 2),
            details=details,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"💥 Erreur prédiction: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Informations sur le modèle chargé"""
    if model is None:
        return {"error": "Modèle non chargé"}
    
    info = {
        "model_type": str(type(model).__name__),
        "model_loaded": True,
        "path_used": model_path_used,
    }
    
    if hasattr(model, 'feature_names_in_'):
        info["features_expected"] = len(model.feature_names_in_)
        info["feature_names"] = model.feature_names_in_.tolist()
    
    if hasattr(model, 'n_features_in_'):
        info["n_features_in"] = model.n_features_in_
        
    return info

@app.get("/test-prediction")
async def test_prediction():
    """Test rapide de prédiction"""
    test_data = InputData(
        surface_reelle_bati=65.0,
        nombre_pieces_principales=3,
        arrondissement=11,
        longitude=2.379,
        latitude=48.858,
        annee_construction_dpe=1980
    )
    
    return await predict_price(test_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)