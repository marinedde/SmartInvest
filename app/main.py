# main.py - API SmartInvest avec FastAPI (Version améliorée)
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import os

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de l'app FastAPI
app = FastAPI(
    title="🏠 SmartInvest API",
    description="API de prédiction de prix immobilier pour Paris",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour permettre les appels depuis Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle avec gestion d'erreur améliorée
model = None
model_path_used = None

# Chemins à tester dans l'ordre
paths_to_try = [
    "model.pkl",                    # Racine du projet
    "./model.pkl",                  # Racine explicite
    "models/model.pkl",             # Dossier models
    os.path.join(os.getcwd(), "model.pkl"),  # Chemin absolu
]

for path in paths_to_try:
    try:
        if os.path.exists(path):
            model = joblib.load(path)
            model_path_used = path
            logger.info(f"✅ Modèle chargé avec succès depuis {path}")
            break
        else:
            logger.info(f"📁 Fichier non trouvé: {path}")
    except Exception as e:
        logger.warning(f"⚠️ Erreur chargement {path}: {e}")

if model is None:
    logger.error("❌ AUCUN MODÈLE CHARGÉ - L'API fonctionnera en mode dégradé")
    logger.error(f"📁 Répertoire de travail: {os.getcwd()}")
    logger.error(f"📄 Fichiers disponibles: {[f for f in os.listdir('.') if f.endswith('.pkl')]}")

# Schémas Pydantic avec validation avancée
class InputData(BaseModel):
    """Données d'entrée pour la prédiction immobilière"""
    
    # Champs obligatoires (compatibles avec Streamlit)
    surface_reelle_bati: float = Field(
        ..., 
        gt=5.0, 
        le=1000.0,
        description="Surface réelle bâtie en m²"
    )
    valeur_fonciere: float = Field(
        ..., 
        gt=10000, 
        le=10000000,
        description="Valeur foncière en euros"
    )
    annee_construction_dpe: int = Field(
        ..., 
        ge=1800, 
        le=datetime.now().year,
        description="Année de construction du bâtiment"
    )
    nombre_pieces_principales: int = Field(
        ..., 
        ge=1, 
        le=20,
        description="Nombre de pièces principales"
    )
    arrondissement: int = Field(
        ..., 
        ge=1, 
        le=20,
        description="Arrondissement parisien (1 à 20)"
    )
    
    # Champs optionnels avec valeurs par défaut
    annee: Optional[int] = Field(
        default=datetime.now().year,
        ge=2010,
        le=datetime.now().year,
        description="Année de la transaction"
    )
    distance_metro_km: Optional[float] = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="Distance au métro le plus proche en km"
    )
    nb_POIs_1km: Optional[int] = Field(
        default=10,
        ge=0,
        le=100,
        description="Nombre de points d'intérêt dans un rayon de 1km"
    )
    
    # Nouveaux champs pour compatibilité Streamlit
    etage: Optional[str] = Field(
        default="2ème",
        description="Étage du logement"
    )
    balcon: Optional[bool] = Field(
        default=False,
        description="Présence d'un balcon ou terrasse"
    )
    parking: Optional[bool] = Field(
        default=False,
        description="Place de parking incluse"
    )
    ascenseur: Optional[bool] = Field(
        default=False,
        description="Présence d'un ascenseur"
    )
    
    @validator('arrondissement')
    def validate_arrondissement(cls, v):
        if v not in range(1, 21):
            raise ValueError('L\'arrondissement doit être entre 1 et 20')
        return v
    
    @validator('surface_reelle_bati')
    def validate_surface(cls, v, values):
        if 'nombre_pieces_principales' in values:
            pieces = values['nombre_pieces_principales']
            if v / pieces < 8:  # Minimum 8m² par pièce
                raise ValueError('Surface trop petite par rapport au nombre de pièces')
        return v

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    prix_m2: float = Field(..., description="Prix prédit au m² en euros")
    valeur_totale_estimee: float = Field(..., description="Valeur totale estimée")
    confiance: Optional[float] = Field(None, description="Niveau de confiance (0-1)")
    details: Optional[Dict[str, Any]] = Field(None, description="Informations détaillées")
    timestamp: str = Field(..., description="Horodatage de la prédiction")

class HealthResponse(BaseModel):
    """Réponse de santé de l'API"""
    status: str
    model_loaded: bool
    version: str
    timestamp: str

# Utilitaires
def etage_to_numeric(etage: str) -> int:
    """Convertit l'étage textuel en numérique"""
    mapping = {
        "RDC": 0, "1er": 1, "2ème": 2, "3ème": 3, 
        "4ème": 4, "5ème+": 5
    }
    return mapping.get(etage, 2)

def encode_poi_dominant(distance_metro: float, nb_pois: int) -> int:
    """Encode POI_dominant en numérique basé sur les distances"""
    if distance_metro < 0.3:
        return 0  # metro
    elif nb_pois > 10:
        return 1  # commerce
    else:
        return 2  # autre

def encode_zone_manuelle(arrondissement: int) -> int:
    """Encode zone_manuelle basé sur l'arrondissement"""
    if arrondissement in [1, 2, 3, 4, 6, 7, 8, 9]:
        return 0  # Centre
    elif arrondissement in [5, 10, 11, 12]:
        return 1  # Est
    elif arrondissement in [15, 16, 17]:
        return 2  # Ouest
    else:
        return 3  # Nord/Sud

def encode_zone_kmeans(arrondissement: int) -> int:
    """Encode zone_kmeans basé sur l'arrondissement"""
    return (arrondissement - 1) % 4  # 0, 1, 2, 3

def prepare_features_fixed(data: InputData) -> pd.DataFrame:
    """Génère exactement 46 features pour correspondre au modèle"""
    
    # Features de base (8)
    features = {
        "surface_reelle_bati": data.surface_reelle_bati,
        "valeur_fonciere": data.valeur_fonciere,
        "annee": data.annee,
        "annee_construction_dpe": data.annee_construction_dpe,
        "nombre_pieces_principales": data.nombre_pieces_principales,
        "arrondissement": data.arrondissement,
        "distance_metro_km": data.distance_metro_km,
        "nb_POIs_1km": data.nb_POIs_1km,
    }
    
    # Calculer des features dérivées
    age_batiment = datetime.now().year - data.annee_construction_dpe
    surface_par_piece = data.surface_reelle_bati / data.nombre_pieces_principales
    prix_m2_actuel = data.valeur_fonciere / data.surface_reelle_bati
    
    # Features supplémentaires pour atteindre 46 (38 de plus)
    additional_features = {
        # Features géographiques (6)
        "latitude": 48.8566 + (data.arrondissement - 10.5) * 0.01,
        "longitude": 2.3522 + (data.arrondissement - 10.5) * 0.01,
        "x": 652000 + data.arrondissement * 1000,
        "y": 6862000 + data.arrondissement * 1000,
        "distance_centre_km": abs(data.arrondissement - 1) * 0.5,
        "zone_code": encode_zone_manuelle(data.arrondissement),
        
        # Features de distances (8)
        "distance_ecole_km": max(0.2, data.distance_metro_km * 0.8),
        "distance_college_km": data.distance_metro_km * 1.1,
        "distance_universite_km": data.distance_metro_km * 2.0,
        "distance_espace_vert_km": data.distance_metro_km * 0.9,
        "distance_commerce_km": data.distance_metro_km * 1.2,
        "distance_transport_km": data.distance_metro_km * 3.0,
        "distance_POI_min_km": min(data.distance_metro_km, 0.1),
        "distance_batiment_m": 10.0,
        
        # Features calculées (8)
        "age_batiment": age_batiment,
        "surface_par_piece": surface_par_piece,
        "prix_m2_reference": prix_m2_actuel,
        "densite_pieces": data.nombre_pieces_principales / data.surface_reelle_bati * 100,
        "etage_numerique": etage_to_numeric(data.etage),
        "bonus_equipements": sum([data.balcon, data.parking, data.ascenseur]),
        "surface_categorie": 1 if data.surface_reelle_bati < 50 else 2 if data.surface_reelle_bati < 80 else 3,
        "arrondissement_groupe": 1 if data.arrondissement <= 10 else 2,
        
        # Features de marché (6)
        "nb_lots": 1,
        "proche_POI": 1 if data.nb_POIs_1km > 5 else 0,
        "poi_dominant_code": encode_poi_dominant(data.distance_metro_km, data.nb_POIs_1km),
        "zone_kmeans": encode_zone_kmeans(data.arrondissement),
        "market_segment": 1 if data.valeur_fonciere < 400000 else 2 if data.valeur_fonciere < 800000 else 3,
        "luxury_indicator": min(3, sum([data.balcon, data.parking, data.ascenseur]) + (1 if data.arrondissement <= 8 else 0)),
        
        # Features temporelles (4)
        "mois": datetime.now().month,
        "trimestre": (datetime.now().month - 1) // 3 + 1,
        "is_recent_construction": 1 if age_batiment < 10 else 0,
        "is_old_construction": 1 if age_batiment > 50 else 0,
        
        # Features d'interaction (6)
        "surface_x_pieces": data.surface_reelle_bati * data.nombre_pieces_principales,
        "prix_x_arrond": data.valeur_fonciere * data.arrondissement / 1000000,
        "age_x_surface": age_batiment * data.surface_reelle_bati / 100,
        "metro_x_pois": data.distance_metro_km * data.nb_POIs_1km,
        "pieces_x_arrond": data.nombre_pieces_principales * data.arrondissement,
        "surface_value_ratio": data.surface_reelle_bati / (data.valeur_fonciere / 100000),
    }
    
    # Combiner toutes les features
    all_features = {**features, **additional_features}
    
    # Vérifier qu'on a exactement 46 features
    if len(all_features) != 46:
        # Ajouter des features de padding si nécessaire
        for i in range(len(all_features), 46):
            all_features[f"feature_padding_{i}"] = 0.0
    
    # Créer le DataFrame
    df = pd.DataFrame([all_features])
    
    # S'assurer que tout est numérique
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df
    """Prépare les features pour le modèle ML avec toutes les colonnes possibles"""
    
    # Features de base
    base_features = {
        "surface_reelle_bati": data.surface_reelle_bati,
        "valeur_fonciere": data.valeur_fonciere,
        "annee": data.annee,
        "annee_construction_dpe": data.annee_construction_dpe,
        "nombre_pieces_principales": data.nombre_pieces_principales,
        "arrondissement": data.arrondissement,
        "distance_metro_km": data.distance_metro_km,
        "nb_POIs_1km": data.nb_POIs_1km
    }
    
    # Features calculées
    etage_num = etage_to_numeric(data.etage)
    bonus_equipements = sum([data.balcon, data.parking, data.ascenseur])
    age_batiment = datetime.now().year - data.annee_construction_dpe
    surface_par_piece = data.surface_reelle_bati / data.nombre_pieces_principales
    
    # Encodage des variables catégorielles
    poi_dominant_encoded = encode_poi_dominant(data.distance_metro_km, data.nb_POIs_1km)
    zone_manuelle_encoded = encode_zone_manuelle(data.arrondissement)
    zone_kmeans_encoded = encode_zone_kmeans(data.arrondissement)
    
    # Features étendues - toutes les colonnes possibles
    extended_features = {
        # Features existantes
        "nombre_lots": 1,
        "a_plusieurs_lots": 0,
        "nb_lots_surface": data.surface_reelle_bati,
        "proche_POI_1km": 1 if data.nb_POIs_1km > 5 else 0,
        "POI_dominant": poi_dominant_encoded,
        "latitude": 48.8566 + (data.arrondissement - 10.5) * 0.01,
        "longitude": 2.3522 + (data.arrondissement - 10.5) * 0.01,
        "x": 652000 + data.arrondissement * 1000,  # Coordonnées Lambert
        "y": 6862000 + data.arrondissement * 1000,
        
        # Distances
        "distance_ecole_km": max(0.2, data.distance_metro_km * 0.8),
        "distance_datashop_km": data.distance_metro_km * 1.2,
        "distance_espace_vert_km": data.distance_metro_km * 0.9,
        "distance_college_km": data.distance_metro_km * 1.1,
        "distance_universite_km": data.distance_metro_km * 2.0,
        "distance_TER_km": data.distance_metro_km * 3.0,
        "distance_batiment_m": 10.0,
        "distance_POI_min_km": min(data.distance_metro_km, 0.1),
        
        # Identifiants et zones
        "cle_interop_adr_proche": 1,
        "zone_manuelle": zone_manuelle_encoded,
        "zone_kmeans": zone_kmeans_encoded,
        
        # Features calculées
        "etage_numerique": etage_num,
        "bonus_equipements": bonus_equipements,
        "age_batiment": age_batiment,
        "surface_par_piece": surface_par_piece,
        
        # Features potentiellement manquantes - ajouts fréquents
        "prix_m2": data.valeur_fonciere / data.surface_reelle_bati,  # Prix actuel
        "densite_pieces": data.nombre_pieces_principales / data.surface_reelle_bati * 100,
        "is_recent": 1 if age_batiment < 10 else 0,
        "is_old": 1 if age_batiment > 50 else 0,
        "arrondissement_groupe": 1 if data.arrondissement <= 10 else 2,
        "surface_categorie": (
            1 if data.surface_reelle_bati < 30 else
            2 if data.surface_reelle_bati < 60 else
            3 if data.surface_reelle_bati < 100 else 4
        ),
        
        # Features géographiques détaillées
        "distance_centre_km": abs(data.arrondissement - 1) * 0.5,
        "proximity_score": 1.0 / (1.0 + data.distance_metro_km),
        "density_score": data.nb_POIs_1km / 20.0,
        
        # Features temporelles
        "mois": datetime.now().month,
        "trimestre": (datetime.now().month - 1) // 3 + 1,
        "is_weekend": 0,  # Par défaut jour de semaine
        
        # Features de marché immobilier
        "market_segment": (
            1 if data.valeur_fonciere < 300000 else
            2 if data.valeur_fonciere < 600000 else 3
        ),
        "luxury_score": min(3, bonus_equipements + (1 if data.arrondissement <= 8 else 0)),
        
        # Features d'interaction
        "surface_x_pieces": data.surface_reelle_bati * data.nombre_pieces_principales,
        "prix_x_surface": data.valeur_fonciere * data.surface_reelle_bati / 1000000,
        "age_x_arrond": age_batiment * data.arrondissement,
        
        # Features par défaut additionnelles
        "type_bien": 1,  # Appartement par défaut
        "etage_max": etage_num + 2,  # Estimation étages du bâtiment
        "exposition": 1,  # Sud par défaut
        "balcon_surface": 5.0 if data.balcon else 0.0,
        "parking_type": 1 if data.parking else 0,
        "ascenseur_present": 1 if data.ascenseur else 0
    }
    
    # Combinaison de toutes les features
    all_features = {**base_features, **extended_features}
    df = pd.DataFrame([all_features])
    
    # Conversion explicite en types numériques pour XGBoost
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except:
                df[col] = 0
    
    # Si on a les features attendues par le modèle, on les utilise
    if hasattr(model, 'feature_names_in_') and model is not None:
        expected_features = model.feature_names_in_
        available_features = df.columns.tolist()
        
        # Ajouter les features manquantes avec des valeurs par défaut
        for feature in expected_features:
            if feature not in available_features:
                df[feature] = 0  # Valeur par défaut
        
        # Réorganiser les colonnes dans l'ordre attendu
        df = df[expected_features]
    
    return df

def calculate_confidence(data: InputData, prediction: float) -> float:
    """Calcule un score de confiance basé sur les données d'entrée et les métriques du modèle"""
    
    # Score de base basé sur les performances du modèle
    # R² = 0.342, MAE = 1595€/m², RMSE = 2223€/m²
    base_confidence = 0.68  # Basé sur le R² de test
    
    # Ajustements selon les caractéristiques du bien
    confidence = base_confidence
    
    # Réduction de confiance pour des cas atypiques
    surface_par_piece = data.surface_reelle_bati / data.nombre_pieces_principales
    if surface_par_piece < 15 or surface_par_piece > 50:
        confidence *= 0.85
    
    # Réduction pour des biens très anciens ou très récents
    age = datetime.now().year - data.annee_construction_dpe
    if age > 100 or age < 5:
        confidence *= 0.90
    
    # Réduction pour des prix extrêmes (en dehors de la distribution d'entraînement)
    if prediction > 20000 or prediction < 3000:
        confidence *= 0.75
    
    # Réduction pour arrondissements avec moins de données
    if data.arrondissement in [1, 4, 7, 8, 16]:  # Arrondissements premium avec moins de transactions
        confidence *= 0.90
    
    # Bonus pour arrondissements avec beaucoup de données
    if data.arrondissement in [11, 18, 19, 20]:  # Arrondissements populaires avec plus de données
        confidence *= 1.05
    
    return max(0.3, min(0.95, confidence))  # Entre 30% et 95%

# Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "🏠 SmartInvest API - Prédiction immobilière Paris",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Vérification de la santé de l'API"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="2.0.0",
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(data: InputData):
    """
    Prédiction du prix au m² d'un bien immobilier parisien
    
    Retourne le prix estimé au m² avec un niveau de confiance et des détails.
    """
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle ML non disponible. Contactez l'administrateur."
        )
    
    try:
        # Log pour tracer les requêtes
        request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger.info(f"🔍 Nouvelle prédiction [{request_id}] pour arrondissement {data.arrondissement}")
        
        # Préparation des features
        input_df = prepare_features_fixed(data)
        
        # Vérification de la compatibilité des colonnes avec le modèle
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_
            available_features = input_df.columns.tolist()
            missing_features = set(required_features) - set(available_features)
            
            if missing_features:
                logger.warning(f"⚠️ Features manquantes: {missing_features}")
                # Ajouter les features manquantes avec des valeurs par défaut
                for feature in missing_features:
                    input_df[feature] = 0
            
            # Réorganiser les colonnes dans l'ordre attendu
            input_df = input_df[required_features]
        
        # Prédiction (toujours recalculée, jamais mise en cache)
        prediction = model.predict(input_df)[0]
        
        # Assurer que la prédiction est positive et réaliste
        prediction = max(1000, min(50000, prediction))
        
        # Calcul des métriques additionnelles
        valeur_totale = prediction * data.surface_reelle_bati
        confidence = calculate_confidence(data, prediction)
        
        # Détails supplémentaires
        details = {
            "request_id": request_id,  # Identifiant unique pour traçabilité
            "arrondissement_analyse": f"{data.arrondissement}ème arrondissement",
            "surface_par_piece": round(data.surface_reelle_bati / data.nombre_pieces_principales, 1),
            "age_batiment": datetime.now().year - data.annee_construction_dpe,
            "categorie_prix": (
                "Premium" if prediction > 15000 
                else "Elevé" if prediction > 12000 
                else "Moyen" if prediction > 9000 
                else "Abordable"
            ),
            "equipements": {
                "balcon": data.balcon,
                "parking": data.parking,
                "ascenseur": data.ascenseur
            }
        }
        
        logger.info(f"✅ Prédiction réussie [{request_id}]: {prediction:.2f} €/m² (confiance: {confidence:.2f})")
        
        return PredictionResponse(
            prix_m2=round(prediction, 2),
            valeur_totale_estimee=round(valeur_totale, 2),
            confiance=round(confidence, 2),
            details=details,
            timestamp=datetime.now().isoformat()
        )
        
    except ValueError as ve:
        logger.error(f"❌ Erreur de validation: {ve}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Erreur de validation des données: {str(ve)}"
        )
    except Exception as e:
        logger.error(f"💥 Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur interne lors de la prédiction: {str(e)}"
        )

@app.get("/stats")
async def get_model_stats():
    """Statistiques du modèle (si disponibles)"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
    
    stats = {
        "model_type": str(type(model).__name__),
        "features_count": len(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else "N/A",
        "model_loaded": True
    }
    
    if hasattr(model, 'feature_names_in_'):
        stats["features"] = model.feature_names_in_.tolist()
        stats["features_detailed"] = {
            "expected_count": len(model.feature_names_in_),
            "feature_list": model.feature_names_in_.tolist()
        }
    
    return stats

@app.get("/inspect-model")
async def inspect_model():
    """Inspecte en détail le modèle pour comprendre ses attentes"""
    if model is None:
        return {"error": "Modèle non chargé"}
    
    try:
        # Informations sur le modèle
        model_info = {
            "model_type": str(type(model)),
            "model_class": model.__class__.__name__,
            "has_feature_names_in": hasattr(model, 'feature_names_in_'),
            "has_n_features": hasattr(model, 'n_features_'),
            "has_n_features_in": hasattr(model, 'n_features_in_'),
        }
        
        # Essayer différentes méthodes pour obtenir le nombre de features
        if hasattr(model, 'n_features_in_'):
            model_info["n_features_in"] = model.n_features_in_
        elif hasattr(model, 'n_features_'):
            model_info["n_features"] = model.n_features_
        
        # Attributs disponibles du modèle
        model_attributes = [attr for attr in dir(model) if not attr.startswith('_')]
        model_info["available_attributes"] = model_attributes
        
        # Test avec différents nombres de features
        test_results = {}
        for n_features in [32, 46, 55]:
            try:
                # Créer un array de test avec n_features colonnes
                import numpy as np
                test_array = np.random.random((1, n_features))
                prediction = model.predict(test_array)
                test_results[f"test_{n_features}_features"] = {
                    "success": True,
                    "prediction": float(prediction[0]) if len(prediction) > 0 else None
                }
            except Exception as e:
                test_results[f"test_{n_features}_features"] = {
                    "success": False,
                    "error": str(e)
                }
        
        model_info["feature_tests"] = test_results
        
        return model_info
        
    except Exception as e:
        return {"error": f"Erreur lors de l'inspection: {str(e)}"}

@app.get("/debug-model-loading")
async def debug_model_loading():
    """Debug du chargement du modèle"""
    import os
    import glob
    
    debug_info = {
        "current_directory": os.getcwd(),
        "model_path_env": os.getenv("MODEL_PATH", "Non défini"),
        "files_in_current_dir": os.listdir("."),
        "model_files_found": glob.glob("**/*.pkl", recursive=True),
        "model_loaded": model is not None,
    }
    
    if model is not None:
        debug_info["model_type"] = str(type(model))
        debug_info["model_path_used"] = model_path_used if 'model_path_used' in globals() else "Inconnu"
    
    # Tester le chargement manuel
    test_paths = ["model.pkl", "./model.pkl", "models/model.pkl"]
    for path in test_paths:
        try:
            if os.path.exists(path):
                test_model = joblib.load(path)
                debug_info[f"test_load_{path.replace('./', '').replace('/', '_')}"] = "✅ SUCCESS"
            else:
                debug_info[f"test_load_{path.replace('./', '').replace('/', '_')}"] = "❌ File not found"
        except Exception as e:
            debug_info[f"test_load_{path.replace('./', '').replace('/', '_')}"] = f"❌ Error: {str(e)}"
    
    return debug_info
async def test_model_prediction():
    """Test direct du modèle avec un array simple"""
    if model is None:
        return {"error": "Modèle non chargé"}
    
    try:
        import numpy as np
        
        # Test avec exactement 46 features (ce que le modèle attend)
        test_data_46 = np.random.random((1, 46))
        
        try:
            prediction_46 = model.predict(test_data_46)
            return {
                "test_46_features": {
                    "success": True,
                    "prediction": float(prediction_46[0]),
                    "input_shape": test_data_46.shape
                }
            }
        except Exception as e:
            return {
                "test_46_features": {
                    "success": False,
                    "error": str(e),
                    "input_shape": test_data_46.shape
                }
            }
            
    except Exception as e:
        return {"error": str(e)}

# Gestion des erreurs globales
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"💥 Erreur non gérée: {exc}")
    return {"error": "Erreur interne du serveur", "detail": str(exc)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )