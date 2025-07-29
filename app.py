# app.py - Version corrigée
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import uuid
import time

# Configuration de la page
st.set_page_config(
    page_title="SmartInvest - Estimation immobilière Paris", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Forcer le rechargement des données à chaque interaction
if 'counter' not in st.session_state:
    st.session_state.counter = 0

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stAlert > div {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<h1 class="main-header">🏠 SmartInvest - Estimation immobilière Paris</h1>', unsafe_allow_html=True)

# Sidebar pour les paramètres avancés
with st.sidebar:
    st.header("⚙️ Paramètres")
    api_url = st.text_input("URL de l'API", value="http://localhost:8001")
    show_debug = st.checkbox("Afficher les détails techniques", value=False)
    
    # Bouton de rafraîchissement manuel
    if st.button("🔄 Forcer le rafraîchissement", help="Force la mise à jour de l'affichage"):
        st.cache_data.clear()
        st.session_state.clear()
        st.rerun()
    
    # Bouton de test rapide
    if st.button("🧪 Test API", help="Teste la connexion à l'API"):
        try:
            test_response = requests.get(f"{api_url}/health", timeout=5)
            if test_response.status_code == 200:
                st.success("✅ API accessible")
            else:
                st.error(f"❌ API erreur {test_response.status_code}")
        except:
            st.error("❌ API non accessible")
    
    st.markdown("---")
    st.markdown("### 📊 Statistiques rapides Paris")
    st.info("Prix moyen Paris : ~10 500 €/m²")
    st.info("Variation annuelle : +3.2%")
    
    # Métriques de performance du modèle
    with st.expander("🧠 Performance du modèle ML"):
        st.markdown("""
        **Métriques sur données de test :**
        - **R² Score** : 0.342 (34.2% de variance expliquée)
        - **MAE** : 1 595 €/m² (erreur absolue moyenne)
        - **RMSE** : 2 223 €/m² (erreur quadratique)
        - **Algorithme** : XGBoost Regressor
        - **Features** : 46 variables explicatives
        
        💡 *Les prédictions incluent une marge d'erreur*
        """)
        
        # Lien vers MLflow si disponible
        if st.button("📈 Voir les détails sur MLflow"):
            st.markdown("[🔗 Accéder à MLflow](https://amaulf-mlflow-server-smartinvest.hf.space/#/experiments/5)")

# Layout principal avec colonnes
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Informations du bien")
    
    # Formulaire principal avec validation
    with st.form("estimation_form"):
        # Groupe 1: Caractéristiques de base
        st.markdown("#### 🏗️ Caractéristiques physiques")
        col_a, col_b = st.columns(2)
        
        with col_a:
            surface = st.number_input(
                "Surface réelle bâtie (m²)", 
                min_value=10.0, 
                max_value=500.0, 
                value=50.0,
                help="Surface habitable réelle du logement"
            )
            pieces = st.slider(
                "Nombre de pièces principales", 
                1, 8, value=2,
                help="Chambres + salon/séjour (hors cuisine, sdb, wc)"
            )
            
        with col_b:
            annee_construction = st.number_input(
                "Année de construction", 
                min_value=1800, 
                max_value=datetime.now().year, 
                value=1970,
                help="Année de construction du bâtiment"
            )
            etage = st.selectbox(
                "Étage",
                ["RDC", "1er", "2ème", "3ème", "4ème", "5ème+"],
                help="Étage du logement"
            )
        
        # Groupe 2: Localisation et valeur
        st.markdown("#### 📍 Localisation et prix")
        col_c, col_d = st.columns(2)
        
        with col_c:
            arrondissement = st.selectbox(
                "Arrondissement (Paris)", 
                list(range(1, 21)),
                index=10,  # 11ème par défaut
                help="Arrondissement de Paris (1er à 20ème)"
            )
            
        with col_d:
            valeur_fonciere = st.number_input(
                "Valeur foncière (€)", 
                min_value=50000, 
                max_value=5000000, 
                value=300000,
                step=10000,
                help="Prix de vente ou estimation actuelle"
            )
        
        # Options avancées (repliables)
        with st.expander("🔧 Options avancées"):
            balcon = st.checkbox("Balcon/Terrasse")
            parking = st.checkbox("Place de parking")
            ascenseur = st.checkbox("Ascenseur")
            
        submit = st.form_submit_button("🔍 Estimer le prix au m²", use_container_width=True)

# Colonne droite pour les résultats
with col2:
    st.subheader("📈 Résultats")

# Traitement de l'estimation - HORS DU FORMULAIRE pour forcer le refresh
if submit:
    # Forcer le rechargement complet
    st.session_state.counter += 1
    
    # Validation des données
    if surface <= 0 or valeur_fonciere <= 0:
        st.error("⚠️ Veuillez vérifier que la surface et la valeur foncière sont positives")
    else:
        # Création d'un identifiant unique pour cette requête
        unique_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Vider complètement le cache
        st.cache_data.clear()
        
        # Créer une clé unique basée sur tous les paramètres
        cache_key = f"{surface}_{pieces}_{arrondissement}_{valeur_fonciere}_{annee_construction}_{etage}_{balcon}_{parking}_{ascenseur}_{unique_id}"
        
        payload = {
            "surface_reelle_bati": surface,
            "annee_construction_dpe": annee_construction,
            "nombre_pieces_principales": pieces,
            "arrondissement": arrondissement,
            "valeur_fonciere": valeur_fonciere,
            "etage": etage,
            "balcon": balcon,
            "parking": parking,
            "ascenseur": ascenseur,
            "request_id": unique_id,
            "cache_buster": cache_key
        }
        
        # Affichage des détails techniques si demandé
        if show_debug:
            with st.expander("🔍 Données envoyées à l'API"):
                st.json(payload)
        
        # Appel à l'API avec headers anti-cache renforcés
        try:
            with st.spinner("🔄 Estimation en cours..."):
                headers = {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
                    "Pragma": "no-cache",
                    "Expires": "Thu, 01 Jan 1970 00:00:00 GMT",
                    "X-Request-ID": unique_id,
                    "X-Cache-Buster": cache_key
                }
                
                response = requests.post(
                    f"{api_url}/predict", 
                    json=payload,
                    headers=headers,
                    timeout=10
                )
                
            # Traitement de la réponse
            if response.status_code == 200:
                result = response.json()
                prix_m2 = result["prix_m2"]
                
                # Affichage des métriques
                st.success("✅ Estimation réalisée avec succès")
                
                # Avertissement sur la précision du modèle
                st.info("📊 **Précision du modèle** : Marge d'erreur moyenne de ±1 595 €/m² (Test MAE)")
                
                # Métriques principales avec intervalles de confiance
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    # Calcul des bornes d'erreur
                    mae = 1595  # MAE du modèle
                    prix_min = max(1000, prix_m2 - mae)
                    prix_max = min(50000, prix_m2 + mae)
                    
                    st.metric(
                        "Prix estimé/m²", 
                        f"{prix_m2:,.0f} €",
                        delta=f"{prix_m2 - 10500:+,.0f} €" if prix_m2 > 0 else None,
                        help=f"Fourchette probable : {prix_min:,.0f} € - {prix_max:,.0f} €/m²"
                    )
                    
                    # Affichage de la fourchette
                    st.caption(f"🎯 **Fourchette :** {prix_min:,.0f} € - {prix_max:,.0f} € /m²")
                    
                with col_m2:
                    valeur_totale = prix_m2 * surface
                    valeur_min = prix_min * surface
                    valeur_max = prix_max * surface
                    
                    st.metric(
                        "Valeur totale estimée", 
                        f"{valeur_totale:,.0f} €",
                        delta=f"{valeur_totale - valeur_fonciere:+,.0f} €",
                        help=f"Fourchette : {valeur_min:,.0f} € - {valeur_max:,.0f} €"
                    )
                    
                    # Affichage de la fourchette
                    st.caption(f"🎯 **Fourchette :** {valeur_min:,.0f} € - {valeur_max:,.0f} €")
                
                # Graphique de comparaison par arrondissement
                st.markdown("#### 📊 Comparaison par arrondissement")
                
                # Données approximatives des prix moyens par arrondissement
                prix_arron = {
                    1: 15500, 2: 13000, 3: 12500, 4: 14000, 5: 12000,
                    6: 14500, 7: 15000, 8: 13500, 9: 11000, 10: 10500,
                    11: 10800, 12: 9500, 13: 8500, 14: 9800, 15: 10200,
                    16: 12500, 17: 11500, 18: 9200, 19: 8800, 20: 9600
                }
                
                df_prix = pd.DataFrame([
                    {"Arrondissement": f"{k}ème", "Prix_moyen": v, "Votre_estimation": prix_m2 if k == arrondissement else None}
                    for k, v in prix_arron.items()
                ])
                
                fig = px.bar(
                    df_prix, 
                    x="Arrondissement", 
                    y="Prix_moyen",
                    title="Prix moyen par arrondissement (€/m²)",
                    color_discrete_sequence=['lightblue']
                )
                
                # Ajout du point pour l'estimation actuelle
                fig.add_scatter(
                    x=[f"{arrondissement}ème"], 
                    y=[prix_m2],
                    mode='markers',
                    marker=dict(size=15, color='red'),
                    name='Votre estimation'
                )
                
                fig.update_layout(height=400, showlegend=True)
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{unique_id}")
                
                # Analyse contextuelle avec marge d'erreur
                st.markdown("#### 🎯 Analyse")
                
                col_analysis1, col_analysis2 = st.columns(2)
                
                with col_analysis1:
                    if prix_m2 > prix_arron[arrondissement]:
                        ecart_pct = ((prix_m2/prix_arron[arrondissement])-1)*100
                        st.info(f"💰 **Au-dessus de la moyenne** du {arrondissement}ème (+{ecart_pct:.1f}%)")
                    else:
                        ecart_pct = ((prix_m2/prix_arron[arrondissement])-1)*100
                        st.info(f"📉 **En-dessous de la moyenne** du {arrondissement}ème ({ecart_pct:.1f}%)")
                
                with col_analysis2:
                    # Fiabilité de l'estimation
                    confiance_pct = result.get("confiance", 0.7) * 100
                    if confiance_pct > 80:
                        st.success(f"🎯 **Confiance élevée** ({confiance_pct:.0f}%)")
                    elif confiance_pct > 60:
                        st.info(f"⚖️ **Confiance modérée** ({confiance_pct:.0f}%)")
                    else:
                        st.warning(f"⚠️ **Confiance limitée** ({confiance_pct:.0f}%)")
                
                # Message sur la marge d'erreur
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                    <strong>📊 À propos de cette estimation (ID: {unique_id}) :</strong><br>
                    • Basée sur un modèle XGBoost entraîné sur les données DVF<br>
                    • Marge d'erreur moyenne : ±1 595 €/m² (68% des cas)<br>
                    • R² Score : 34.2% (performance sur données de test)<br>
                    • Utiliser comme indication, pas comme évaluation définitive
                </div>
                """, unsafe_allow_html=True)
                
                # Informations complémentaires en bas
                st.markdown("---")
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.markdown("**🏗️ Age du bien**")
                    age = datetime.now().year - annee_construction
                    if age < 10:
                        st.success(f"Récent ({age} ans)")
                    elif age < 30:
                        st.info(f"Moderne ({age} ans)")
                    else:
                        st.warning(f"Ancien ({age} ans)")
                
                with col_info2:
                    st.markdown("**📐 Ratio surface/pièces**")
                    ratio = surface / pieces
                    st.metric("m²/pièce", f"{ratio:.1f}")
                
                with col_info3:
                    st.markdown("**💡 Conseil**")
                    if prix_m2 > 12000:
                        st.info("Secteur premium")
                    elif prix_m2 > 9000:
                        st.info("Bon rapport qualité/prix")
                    else:
                        st.success("Secteur abordable")
                        
            else:
                error_detail = response.json().get('detail', 'Erreur inconnue') if response.headers.get('content-type') == 'application/json' else response.text
                st.error(f"❌ Erreur de l'API ({response.status_code}): {error_detail}")
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Timeout: L'API met trop de temps à répondre")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Erreur de connexion: Vérifiez que l'API est démarrée et accessible")
        except requests.exceptions.RequestException as e:
            st.error(f"🚫 Erreur lors de l'appel à l'API: {e}")
        except Exception as e:
            st.error(f"💥 Erreur inattendue: {e}")

# Footer avec informations
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    📱 SmartInvest - Estimation immobilière intelligente | 
    🔒 Données sécurisées | 
    📊 Basé sur les données DVF (Demandes de Valeurs Foncières)
</div>
""", unsafe_allow_html=True)