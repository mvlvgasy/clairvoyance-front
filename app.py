import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Clairvoyance AI - Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Récupération de l'URL depuis les secrets
# Assurez-vous que le secret est bien configuré sur Streamlit Cloud
try:
    API_URL = st.secrets["API_URL"]
except:
    st.error("L'URL de l'API n'est pas configurée dans les secrets.")
    API_URL = "https://clairvoyance-api-yolov8-mutliclass-v2-sans-lifesp-jhcqfzuifa-ew.a.run.app/"

# --- STYLE CSS ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.1rem;
    }
    /* Petites bordures pour séparer les colonnes visuellement */
    div[data-testid="column"] {
        padding: 10px;
        border-radius: 5px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION D'ENVOI ---
def send_image_to_api(image_bytes, endpoint):
    """Envoie l'image à l'API et récupère la réponse JSON"""
    if not API_URL:
        return None
    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{API_URL}/{endpoint}", files=files)
        if response.status_code == 200:
            return response
        else:
            st.error(f"❌ Erreur API ({endpoint}): {response.status_code}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur connexion ({endpoint}): {e}")
        return None

# ==========================================
# EN-TÊTE & UPLOAD
# ==========================================
st.title("🔮 Clairvoyance : L'Évolution du Modèle")
st.markdown("### *Du prototype pédagogique à la performance industrielle*")

uploaded_file = st.file_uploader("Chargez une image pour tester l'évolution des 3 architectures", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Préparation de l'image
    image = Image.open(uploaded_file)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    bytes_data = img_bytes.getvalue()

    with st.expander("📸 Voir l'image originale", expanded=False):
        st.image(image, caption="Image Source", use_container_width=True)

    # BOUTON D'ACTION
    if st.button("LANCER L'ANALYSE TEMPORELLE 🚀"):
        st.balloons()

        # Organisation : Passé (CNN) -> Présent (TRUSF) -> Futur (YOLOv8)
        col_past, col_present, col_future = st.columns(3, gap="medium")

        # ==========================================
        # 1. LE PASSÉ : CNN "Naïf"
        # ==========================================
        with col_past:
            st.header("1️⃣ Le Passé")
            st.subheader("Approche Naïve (CNN)")
            st.caption("Classification simple d'image")

            st.markdown("---")

            with st.spinner('Analyse CNN...'):
                response_cnn = send_image_to_api(bytes_data, "predict")

                if response_cnn:
                    data = response_cnn.json()
                    st.info(f"Prédiction : **{data['prediction'].upper()}**")
                    st.metric("Confiance", f"{data['confidence']:.2%}")

                    st.markdown("**Limites :**")
                    st.caption("Ne voit qu'un seul véhicule. Échoue sur le trafic dense.")

                    st.write("Probabilités :")
                    st.bar_chart(data['all_probabilities'], height=150)
                else:
                    st.warning("Service CNN indisponible")

        # ==========================================
        # 2. LE PRÉSENT : TRUSF "Custom"
        # ==========================================
        with col_present:
            st.header("2️⃣ Le Présent")
            st.subheader("Architecture TRUSF")
            st.caption("Modèle Propriétaire (Custom)")

            st.markdown("---")

            # Ici, on présente le concept architectural (Notre travail actuel)
            st.info("🛠️ **Architecture en développement**")

            try:
                # Affichage de l'image "Plan IKEA" pour illustrer la construction
                st.image("plan_ikea.jpg", caption="Conception couche par couche", use_container_width=True)
            except:
                st.write("Plan architectural non chargé.")

            st.markdown("""
            **Notre Démarche :**
            Plutôt que d'utiliser une boîte noire, nous reconstruisons la logique de détection :
            * **Backbone :** Darknet-53
            * **Neck :** FPN (Pyramide de features)
            * **Head :** Régression des Bbox
            """)

            st.warning("Résultats : Prometteurs mais en cours d'optimisation.")

        # ==========================================
        # 3. LE FUTUR : YOLOv8 "SOTA"
        # ==========================================
        with col_future:
            st.header("3️⃣ Le Futur")
            st.subheader("Performance SOTA")
            st.caption("Standard Industriel (YOLOv8)")

            st.markdown("---")

            with st.spinner('Inférence SOTA...'):
                response_yolo = send_image_to_api(bytes_data, "predict_yolo_image")

                if response_yolo:
                    data_yolo = response_yolo.json()

                    # 1. Image Détectée
                    try:
                        b64_string = data_yolo['image_data']['b64']
                        img_decoded = base64.b64decode(b64_string)
                        det_img = Image.open(io.BytesIO(img_decoded))
                        st.image(det_img, caption="Détection Haute Précision", use_container_width=True)
                    except:
                        st.error("Erreur d'affichage image")

                    # 2. KPIs de performance
                    st.success(f"⚡ Vitesse : **{data_yolo['performance']['inference']:.1f} ms**")

                    counts = data_yolo['summary']
                    c1, c2 = st.columns(2)
                    c1.metric("Véhicules Légers", counts.get('car', 0) + counts.get('motorcycle', 0))
                    c2.metric("Poids Lourds", counts.get('truck', 0) + counts.get('bus', 0))

                    # 3. Données
                    if data_yolo.get('detections'):
                        with st.expander("📊 Données Brutes"):
                            df = pd.DataFrame(data_yolo['detections'])
                            st.dataframe(df[['label', 'confidence']], use_container_width=True, hide_index=True)

# Pied de page
st.markdown("---")
st.markdown("<div style='text-align: center; color: grey;'>Equipe Clairvoyance © 2025 - Data Science Project</div>", unsafe_allow_html=True)
