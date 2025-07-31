# app.py - Version corrigée selon vraies features du modèle
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

# CSS personnalisé
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
</style>
""", unsafe_allow_html=True)

# En-tête principal
st.markdown('<h1 class="main-header">🏠 SmartInvest - Estimation immobilière Paris</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    api_url = st.text_input("URL de l'API", value="http://localhost:8001")
    show_debug = st.checkbox("Afficher les détails techniques", value=False)
    
    # Test API
    if st.button("🧪 Test API"):
        try:
            test_response = requests.get(f"{api_url}/health", timeout=5)
            if test_response.status_code == 200:
                st.success("✅ API accessible")
            else:
                st.error(f"❌ API erreur {test_response.status_code}")
        except:
            st.error("❌ API non accessible")
    
    st.markdown("---")
    st.markdown("### 📊 Performance du modèle")
    st.info("MAE : 1,596 €/m²")
    st.info("MAPE : 14.9%")
    st.info("Features : 20 variables")

# Layout principal
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Caractéristiques du bien")
    
    # ⚠️ AVERTISSEMENT pour utilisateur
    st.warning("⚠️ **Important** : Cette application prédit le prix au m² basé uniquement sur les caractéristiques disponibles dans les données DVF officielles.")
    
    with st.form("estimation_form"):
        # Caractéristiques de base (DVF)
        st.markdown("#### 🏗️ Caractéristiques principales")
        col_a, col_b = st.columns(2)
        
        with col_a:
            surface = st.number_input(
                "Surface réelle bâtie (m²)", 
                min_value=10.0, 
                max_value=500.0, 
                value=50.0,
                help="Surface habitable du logement"
            )
            pieces = st.slider(
                "Nombre de pièces principales", 
                1, 8, value=2,
                help="Chambres + salon (hors cuisine, sdb)"
            )
            
        with col_b:
            nombre_lots = st.number_input(
                "Nombre de lots", 
                min_value=1, 
                max_value=10, 
                value=1,
                help="Nombre de lots dans la copropriété"
            )
            annee = st.selectbox(
                "Année de transaction",
                [2020, 2021, 2022, 2023, 2024],
                index=4,
                help="Année de référence pour l'estimation"
            )
        
        # Localisation (DVF)
        st.markdown("#### 📍 Localisation")
        arrondissement = st.selectbox(
            "Arrondissement (Paris)", 
            list(range(1, 21)),
            index=10,
            help="Arrondissement de Paris (1er à 20ème)"
        )
        
        # Coordonnées approximatives (optionnel)
        with st.expander("🗺️ Localisation précise (optionnel)"):
            st.info("💡 Les coordonnées amélioreront la précision de l'estimation")
            col_coord1, col_coord2 = st.columns(2)
            with col_coord1:
                longitude = st.number_input(
                    "Longitude", 
                    min_value=2.224, 
                    max_value=2.469, 
                    value=2.347,  # Centre de Paris
                    format="%.6f",
                    help="Coordonnée Est (optionnel mais recommandé)"
                )
            with col_coord2:
                latitude = st.number_input(
                    "Latitude", 
                    min_value=48.815, 
                    max_value=48.902, 
                    value=48.857,  # Centre de Paris
                    format="%.6f",
                    help="Coordonnée Nord (optionnel mais recommandé)"
                )
        
        # Features enrichies disponibles (selon votre modèle)
        with st.expander("🎯 Features enrichies automatiques"):
            st.markdown("""
            **Ces données sont calculées automatiquement selon la localisation :**
            - 🚇 Distance aux stations de métro
            - 🏫 Distance aux écoles et universités  
            - 🌳 Distance aux espaces verts
            - 🏢 Distance aux points d'intérêt
            - 🏠 Distance aux bâtiments proches
            - 📅 Année de construction (base DPE)
            """)
        
        submit = st.form_submit_button("🔍 Estimer le prix au m²", use_container_width=True)

# Colonne droite pour les résultats
with col2:
    st.subheader("📈 Estimation")

# Traitement SANS valeur foncière
if submit:
    if surface <= 0:
        st.error("⚠️ Veuillez vérifier la surface")
    else:
        unique_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        # Payload selon VOS vraies features
        payload = {
            # Features de base DVF
            "surface_reelle_bati": surface,
            "nombre_pieces_principales": pieces,
            "longitude": longitude,
            "latitude": latitude,
            "nombre_lots": nombre_lots,
            "annee": annee,
            "arrondissement": arrondissement,
            
            # Features calculées automatiquement par l'API
            # (l'API devra calculer ces distances selon longitude/latitude)
            # "distance_metro_km": calculé par API,
            # "distance_ecole_km": calculé par API,
            # "distance_espace_vert_km": calculé par API,
            # etc...
            
            "request_id": unique_id
        }
        
        if show_debug:
            with st.expander("🔍 Données envoyées"):
                st.json(payload)
        
        try:
            with st.spinner("🔄 Calcul en cours..."):
                headers = {
                    "Content-Type": "application/json",
                    "Cache-Control": "no-cache",
                    "X-Request-ID": unique_id
                }
                
                response = requests.post(
                    f"{api_url}/predict", 
                    json=payload,
                    headers=headers,
                    timeout=10
                )
                
            if response.status_code == 200:
                result = response.json()
                prix_m2 = result["prix_m2"]
                
                st.success("✅ Estimation réalisée")
                
                # Métriques principales
                col_m1, col_m2 = st.columns(2)
                
                with col_m1:
                    mae = 1596  # Votre MAE réelle
                    prix_min = max(1000, prix_m2 - mae)
                    prix_max = min(50000, prix_m2 + mae)
                    
                    st.metric(
                        "Prix estimé/m²", 
                        f"{prix_m2:,.0f} €",
                        help=f"Fourchette : {prix_min:,.0f} - {prix_max:,.0f} €/m²"
                    )
                    
                    st.caption(f"🎯 **Marge d'erreur :** ±{mae:,.0f} €/m²")
                    
                with col_m2:
                    valeur_totale = prix_m2 * surface
                    st.metric(
                        "Valeur totale estimée", 
                        f"{valeur_totale:,.0f} €"
                    )
                    
                    prix_moyen_arr = {
                        1: 15500, 2: 13000, 3: 12500, 4: 14000, 5: 12000,
                        6: 14500, 7: 15000, 8: 13500, 9: 11000, 10: 10500,
                        11: 10800, 12: 9500, 13: 8500, 14: 9800, 15: 10200,
                        16: 12500, 17: 11500, 18: 9200, 19: 8800, 20: 9600
                    }
                    
                    ecart_pct = ((prix_m2/prix_moyen_arr[arrondissement])-1)*100
                    if ecart_pct > 0:
                        st.metric("vs Moyenne arrond.", f"+{ecart_pct:.1f}%")
                    else:
                        st.metric("vs Moyenne arrond.", f"{ecart_pct:.1f}%")
                
                # Graphique de comparaison
                st.markdown("#### 📊 Comparaison")
                
                df_prix = pd.DataFrame([
                    {"Arrond": f"{k}e", "Prix_moyen": v, "Type": "Moyenne marché"}
                    for k, v in prix_moyen_arr.items()
                ])
                
                # Ajouter votre estimation
                df_estimation = pd.DataFrame([{
                    "Arrond": f"{arrondissement}e", 
                    "Prix_moyen": prix_m2, 
                    "Type": "Votre estimation"
                }])
                
                df_combined = pd.concat([df_prix, df_estimation])
                
                fig = px.bar(
                    df_combined, 
                    x="Arrond", 
                    y="Prix_moyen",
                    color="Type",
                    title="Comparaison avec le marché (€/m²)",
                    color_discrete_map={
                        "Moyenne marché": "lightblue",
                        "Votre estimation": "red"
                    }
                )
                
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Analyse contextuelle
                st.markdown("#### 🎯 Analyse")
                
                if prix_m2 > prix_moyen_arr[arrondissement] * 1.1:
                    st.info(f"💰 **Bien premium** dans le {arrondissement}e (+{ecart_pct:.1f}%)")
                elif prix_m2 < prix_moyen_arr[arrondissement] * 0.9:
                    st.success(f"💡 **Opportunité potentielle** dans le {arrondissement}e ({ecart_pct:.1f}%)")
                else:
                    st.info(f"⚖️ **Prix de marché** dans le {arrondissement}e ({ecart_pct:.1f}%)")
                
                # Informations sur la précision
                st.markdown(f"""
                <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px;'>
                    <strong>📊 Fiabilité de l'estimation :</strong><br>
                    • Modèle XGBoost (MAE: 1,596 €/m²)<br>
                    • Basé sur {surface}m², {pieces} pièces, {arrondissement}e arr.<br>
                    • Features enrichies calculées automatiquement<br>
                    • ID: {unique_id}
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.error(f"❌ Erreur API ({response.status_code})")
                
        except Exception as e:
            st.error(f"🚫 Erreur : {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.8em;'>
    📱 SmartInvest | 🏗️ Basé sur données DVF officielles | 🧠 ML avec features enrichies
</div>
""", unsafe_allow_html=True)