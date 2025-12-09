import streamlit as st
import requests
from PIL import Image
import io
import base64
import pandas as pd

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Clairvoyance AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Url api de service
API_URL = "https://clairvoyance-api-yolov8-mutliclass-v2-401633208612.europe-west1.run.app/"

API_URL = st.secrets["API_URL"]

# --- STYLE CSS (Minimaliste - On garde le gris par défaut) ---
st.markdown("""
    <style>
    /* On garde juste le style des boutons pour qu'ils soient beaux */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    /* On agrandit un peu les chiffres */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTION D'ENVOI ---
def send_image_to_api(image_bytes, endpoint):
    """Envoie l'image à l'API et récupère la réponse JSON"""
    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{API_URL}/{endpoint}", files=files)

        if response.status_code == 200:
            return response
        else:
            st.error(f"❌ Erreur API ({response.status_code}): {response.text}")
            return None
    except Exception as e:
        st.error(f"❌ Erreur de connexion : {e}")
        return None

# ==========================================
# PAGE 1 : CLASSIFICATION (CNN)
# ==========================================
def page_classification():
    st.title("1️⃣ Classification Simple (CNN)")
    st.markdown("### *L'approche initiale : Identifier le véhicule dominant*")

    col_upload, col_result = st.columns([1, 1.5])

    with col_upload:
        st.info("Ce modèle (CNN 256px) est entraîné pour classifier une image contenant **un seul véhicule majoritaire**.")
        uploaded_file = st.file_uploader("Choisissez une image", type=['jpg', 'jpeg', 'png'], key="cnn")

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Aperçu", use_container_width=True)

    with col_result:
        if uploaded_file and st.button("Lancer l'analyse CNN 🧠"):
            with st.spinner('Analyse des pixels en cours...'):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG')

                response = send_image_to_api(img_bytes.getvalue(), "predict")

                if response:
                    # 🎈 LES BALLONS ICI 🎈
                    st.balloons()

                    data = response.json()
                    pred = data['prediction']
                    conf = data['confidence']
                    probs = data['all_probabilities']

                    st.success(f"### Résultat : {pred.upper()} 🎯")
                    st.metric("Niveau de confiance", f"{conf:.2%}")

                    st.write("**Répartition des probabilités :**")
                    st.bar_chart(probs)

# ==========================================
# PAGE 2 : DÉTECTION SOTA (YOLOv8)
# ==========================================
def page_yolov8():
    st.title("2️⃣ Détection SOTA (YOLOv8)")

    # KPI STATIQUES
    with st.container():
        st.markdown("### 📊 Performances Globales du Modèle")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("mAP@50", "92.4%", "+1.2%")
        k2.metric("Précision", "89.3%", "Global")
        k3.metric("Vitesse Moyenne", "140ms", "Inférence")
        k4.metric("Architecture", "Nano", "6.2 MB")

    st.divider()

    st.markdown("### 🎯 Test en Temps Réel")
    uploaded_file = st.file_uploader("Chargez une photo de trafic complexe", type=['jpg', 'jpeg', 'png'], key="yolo")

    if uploaded_file is not None:
        col_orig, col_pred = st.columns(2)

        # Image Originale
        image = Image.open(uploaded_file)
        col_orig.image(image, caption="Image Originale", use_container_width=True)

        if st.button("Lancer la Détection YOLO 🚀"):
            with st.spinner('Transmission au Cloud Run...'):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG')

                response = send_image_to_api(img_bytes.getvalue(), "predict_yolo_image")

                if response:
                    # 🎈 LES BALLONS ICI AUSSI 🎈
                    st.balloons()

                    data = response.json()

                    # A. Décodage Image
                    try:
                        b64_string = data['image_data']['b64']
                        img_decoded = base64.b64decode(b64_string)
                        detected_image = Image.open(io.BytesIO(img_decoded))

                        col_pred.image(detected_image, caption="Détection IA", use_container_width=True)
                        col_pred.success("✅ Traitement terminé !")
                    except Exception as e:
                        st.error(f"Erreur décodage : {e}")

                    # B. Stats Dynamiques
                    st.markdown("#### ⚡ Statistiques de cette inférence")

                    real_speed = data['performance']['inference']
                    counts = data['summary']

                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Vitesse Réelle", f"{real_speed:.1f} ms")
                    s2.metric("Voitures 🚗", counts.get('car', 0))
                    s3.metric("Motos 🏍️", counts.get('motorcycle', 0))
                    s4.metric("Lourds 🚛", counts.get('truck', 0) + counts.get('bus', 0))

                    # C. Tableau
                    if data.get('detections'):
                        with st.expander("📋 Voir le détail des boîtes"):
                            df = pd.DataFrame(data['detections'])
                            st.dataframe(
                                df[['label', 'confidence', 'bbox']].style.format({"confidence": "{:.2%}"}),
                                use_container_width=True
                            )

# ==========================================
# PAGE 3 : LE PROJET TRUSF (YOLOv3 Custom)
# ==========================================
def page_trusf():
    st.title("3️⃣ Projet TRUSF (Custom YOLOv3)")
    st.markdown("### *Tu Regardes Une Seule Fois - Architecture Maison*")

    st.warning("🚧 **Zone de Construction : Modèle en cours d'assemblage**")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### L'approche 'IKEA'")
        st.write("""
        Pour maîtriser la Computer Vision, nous ne voulions pas juste utiliser une librairie.
        Nous avons reconstruit YOLO couche par couche :
        1.  **Backbone :** Extraction de features.
        2.  **Neck :** Mélange des échelles (FPN).
        3.  **Head :** Prédiction des Bounding Boxes.
        """)
    with col2:
        st.image("/home/aurelien/code/baptperr/clairvoyance-front/Instructions d'assemblages_page-0001.jpg", caption="Concept d'assemblage TRUSF")

    st.markdown("---")
    st.markdown("#### 🎥 Démo Technique")
    # st.video("https://www.youtube.com/watch?v=MPV2METPeJU")


# ==========================================
# NAVIGATION
# ==========================================
st.sidebar.image("https://emojigraph.org/media/apple/crystal-ball_1f52e.png", width=80)
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Aller vers :",
    ["1. Classification (CNN)", "2. Détection (YOLOv8)", "3. Projet TRUSF"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Equipe Clairvoyance © 2025")

if page == "1. Classification (CNN)":
    page_classification()
elif page == "2. Détection (YOLOv8)":
    page_yolov8()
elif page == "3. Projet TRUSF":
    page_trusf()
