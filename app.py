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

# --- FONCTION MAGIQUE POUR L'ARRIÈRE-PLAN ---
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Tente de charger l'image de fond (si elle existe)
try:
    img = get_img_as_base64("background.png")

    # CSS AVANCÉ : BACKGROUND + EFFET VERRE (GLASSMORPHISM)
    page_bg_img = f"""
    <style>
    /* 1. L'IMAGE DE FOND (Générale) */
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{img}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* 2. LE CONTENEUR PRINCIPAL (Le "Gros Carré") */
    /* C'est ici qu'on crée la zone de lecture */
    .block-container {{
        background-color: rgba(15, 15, 15, 0.85); /* Noir à 85% d'opacité */
        backdrop-filter: blur(8px); /* Effet de flou sur l'image derrière */
        border-radius: 20px; /* Bords arrondis */
        padding: 3rem !important; /* Marge intérieure */
        margin-top: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); /* Ombre portée pour le relief */
        border: 1px solid rgba(255, 255, 255, 0.1); /* Fine bordure brillante */
    }}

    /* 3. COULEURS DU TEXTE */
    /* On force tout en blanc pour le contraste sur le fond noir */
    h1, h2, h3, h4, h5, p, span, div, label {{
        color: #ffffff !important;
    }}

    /* 4. STYLE DES BOUTONS */
    .stButton>button {{
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-image: linear-gradient(to right, #FF4B4B, #FF914D); /* Dégradé stylé */
        color: white;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4); /* Lueur sous le bouton */
    }}

    /* 5. ZONE DE DÉPÔT DE FICHIER (File Uploader) */
    [data-testid="stFileUploader"] {{
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

except FileNotFoundError:
    pass


# Récupération de l'URL API
try:
    API_URL = st.secrets["API_URL"]
except:
    API_URL = ""

# --- FONCTION D'ENVOI ---
def send_image_to_api(image_bytes, endpoint):
    if not API_URL:
        return None
    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{API_URL}/{endpoint}", files=files)
        if response.status_code == 200:
            return response
        return None
    except:
        return None

# ==========================================
# EN-TÊTE (LOGO + TITRE)
# ==========================================

# On crée 2 colonnes : une petite pour le logo, une grande pour le titre
col_logo, col_title = st.columns([1, 6])

with col_logo:
    try:
        # Affiche le logo s'il existe
        st.image("logo2.png", use_container_width=True)
    except:
        st.write("🔮") # Emoji si pas de logo

with col_title:
    st.title("Clairvoyance : L'Évolution")
    st.markdown("### *Du prototype pédagogique à la performance industrielle*")

# ==========================================
# UPLOAD
# ==========================================
uploaded_file = st.file_uploader("Chargez une image pour tester l'évolution des 3 architectures", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    bytes_data = img_bytes.getvalue()

    with st.expander("📸 Voir l'image originale"):
        st.image(image, caption="Image Source", use_container_width=True)

    if st.button("LANCER L'ANALYSE TEMPORELLE 🚀"):
        st.balloons()

        # 3 Colonnes
        col_past, col_present, col_future = st.columns(3, gap="medium")

        # --- 1. LE PASSÉ (CNN) ---
        with col_past:
            st.header("1️⃣ Le Passé")
            st.caption("Approche Naïve (CNN)")
            st.markdown("---")

            with st.spinner('Analyse CNN...'):
                response = send_image_to_api(bytes_data, "predict")
                if response:
                    data = response.json()
                    st.info(f"Prédiction : **{data['prediction'].upper()}**")
                    st.metric("Confiance", f"{data['confidence']:.2%}")
                    st.bar_chart(data['all_probabilities'], height=150)
                else:
                    st.warning("Service indisponible")

        # --- 2. LE PRÉSENT (TRUSF) ---
        with col_present:
            st.header("2️⃣ Le Présent")
            st.caption("Architecture TRUSF (Maison)")
            st.markdown("---")

            st.info("🛠️ **Architecture en développement**")
            try:
                st.image("plan_ikea.jpg", caption="Concept", use_container_width=True)
            except:
                pass

            st.markdown("""
            **Logique :** Reconstruction couche par couche (Backbone > Neck > Head) pour maîtriser la détection.
            """)

        # --- 3. LE FUTUR (YOLO SOTA) ---
        with col_future:
            st.header("3️⃣ Le Futur")
            st.caption("Standard Industriel (YOLOv8)")
            st.markdown("---")

            with st.spinner('Inférence SOTA...'):
                response = send_image_to_api(bytes_data, "predict_yolo_image")
                if response:
                    data = response.json()
                    try:
                        b64_string = data['image_data']['b64']
                        img_decoded = base64.b64decode(b64_string)
                        st.image(Image.open(io.BytesIO(img_decoded)), caption="Détection", use_container_width=True)
                    except:
                        pass

                    st.success(f"⚡ {data['performance']['inference']:.1f} ms")

                    counts = data['summary']
                    c1, c2 = st.columns(2)
                    c1.metric("Légers", counts.get('car', 0) + counts.get('motorcycle', 0))
                    c2.metric("Lourds", counts.get('truck', 0) + counts.get('bus', 0))

st.markdown("---")
st.markdown("<div style='text-align: center;'>Equipe Clairvoyance © 2025</div>", unsafe_allow_html=True)
