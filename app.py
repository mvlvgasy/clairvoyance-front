import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import pandas as pd
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Clairvoyance AI - Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- COULEURS POUR DESSIN FRONTEND (Modèle Maison) ---
CLASS_COLORS_FRONT = {
    "Car": "#FF0000",       # Rouge
    "Bus": "#0000FF",       # Bleu
    "Truck": "#00FF00",     # Vert
    "Motorcycle": "#FFFF00" # Jaune
}

# --- GESTION ROBUSTE DE L'ARRIÈRE-PLAN ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Recherche automatique de l'image de fond
bg_file_path = None
for file in ["background.jpg", "background.png", "background.jpeg", "Capture d'écran 2025-12-11 101557.jpg"]:
    if os.path.exists(file):
        bg_file_path = file
        break

if bg_file_path:
    try:
        img_b64 = get_base64_of_bin_file(bg_file_path)
        ext = "png" if bg_file_path.lower().endswith(".png") else "jpeg"

        page_bg_img = f"""
        <style>
        /* 1. L'IMAGE DE FOND */
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/{ext};base64,{img_b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* 2. LE CADRE (GLASSMORPHISM) */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.80);
            backdrop-filter: blur(5px);
            border-radius: 15px;
            padding: 3rem !important;
            margin-top: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}

        /* 3. TEXTE BLANC PAR DEFAUT */
        h1, h2, h3, h4, h5, p, span, div, label, li {{
            color: #ffffff !important;
        }}

        /* 4. BOUTONS */
        .stButton>button {{
            width: 100%;
            border-radius: 8px;
            height: 3em;
            background-image: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
            color: white;
            font-weight: bold;
            border: none;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            transform: scale(1.02);
            box-shadow: 0 6px 15px rgba(255, 75, 75, 0.5);
        }}

        /* 5. ZONE D'UPLOAD */
        [data-testid="stFileUploader"] {{
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
        }}

        /* 6. MODIF CNN : Classe en Gros */
        .big-pred {{
            font-size: 45px !important;
            font-weight: 800 !important;
            color: #FF4B4B !important; /* Rouge flash */
            text-transform: uppercase;
            text-align: center;
            margin-bottom: 0px;
        }}
        .pred-label {{
            font-size: 18px !important;
            color: #cccccc !important;
            text-align: center;
            margin-bottom: -10px;
        }}

        /* 7. NOUVEAUX TITRES COLONNES */
        .col-header-small {{
            font-size: 14px !important;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: #aaaaaa !important;
            margin-bottom: -5px;
        }}
        .col-header-big {{
            font-size: 32px !important;
            font-weight: 700 !important;
            color: white !important;
            margin-top: 0px;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Erreur chargement fond : {e}")
else:
    pass


# Récupération de l'URL API
try:
    API_URL = st.secrets["API_URL"]
except:
    API_URL = "https://clairvoyance-api-yolov8-mutliclass-vraifinal-401633208612.europe-west1.run.app"

# --- FONCTION D'ENVOI ---
def send_image_to_api(image_bytes, endpoint):
    if not API_URL:
        st.error("URL API manquante.")
        return None
    try:
        files = {'file': ('image.jpg', image_bytes, 'image/jpeg')}
        response = requests.post(f"{API_URL}/{endpoint}", files=files, timeout=120)
        if response.status_code == 200:
            return response
        return None
    except:
        return None

# ==========================================
# EN-TÊTE
# ==========================================

col_logo, col_title = st.columns([1, 6])

with col_logo:
    logo_path = "logo2.png" if os.path.exists("logo2.png") else None
    if logo_path:
        st.image(logo_path, width="stretch")
    else:
        st.write("🔮")

with col_title:
    st.title("Clairvoyance : L’œil du véhicule autonome")
    st.markdown("### *Comparatif des architectures de vision par ordinateur*")

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
        st.image(image, caption="Image Source", width="stretch")

    if st.button("LANCER L'ANALYSE TEMPORELLE 🚀"):
        st.balloons()

        # 3 Colonnes
        col_past, col_present, col_future = st.columns(3, gap="medium")

        # ==========================================
        # 1. LE PASSÉ (CNN)
        # ==========================================
        with col_past:
            # TITRES MODIFIÉS
            st.markdown('<div class="col-header-small">1 - LE PASSÉ</div>', unsafe_allow_html=True)
            st.markdown('<div class="col-header-big">CNN Naïf</div>', unsafe_allow_html=True)
            st.markdown("---")

            with st.spinner('Analyse CNN...'):
                response = send_image_to_api(bytes_data, "predict")
                if response:
                    data = response.json()

                    # --- AFFICHAGE CNN MODIFIÉ (Gros Texte) ---
                    pred_class = data['prediction']
                    st.markdown(f"""
                        <div class="pred-label">Prédiction :</div>
                        <div class="big-pred">{pred_class}</div>
                        <br>
                    """, unsafe_allow_html=True)

                    st.metric("Niveau de Confiance", f"{data['confidence']:.2%}")

                    st.write("Répartition :")
                    st.bar_chart(data['all_probabilities'], height=150)
                else:
                    st.warning("Service indisponible")

        # ==========================================
        # 2. LE PRÉSENT (TRUSF - YOLO MAISON)
        # ==========================================
        with col_present:
            # TITRES MODIFIÉS
            st.markdown('<div class="col-header-small">2 - LE PRÉSENT</div>', unsafe_allow_html=True)
            st.markdown('<div class="col-header-big">Modèle TRUSF</div>', unsafe_allow_html=True)
            st.markdown("---")

            with st.spinner('Inférence Custom YOLO...'):
                response = send_image_to_api(bytes_data, "predict_custom_yolo")

                if response:
                    data = response.json()

                    # --- DESSIN FRONTEND (FORCE TEXTE NOIR) ---
                    if data.get('detections'):
                        try:
                            # On copie l'image originale
                            img_draw = image.copy()
                            draw = ImageDraw.Draw(img_draw)

                            # Pour le texte, on essaie de charger une police par défaut, sinon fallback
                            try:
                                font = ImageFont.load_default()
                            except:
                                font = None

                            for det in data['detections']:
                                bbox = det['bbox']
                                label = det['label']
                                conf = det['confidence']

                                # Couleur de la boite
                                color_hex = CLASS_COLORS_FRONT.get(label, "#FF0000")

                                # Dessin Boite
                                draw.rectangle(bbox, outline=color_hex, width=4)

                                # Préparation Texte (Label + %)
                                text_str = f"{label} {conf:.0%}"

                                # Fond du texte (petit rectangle pour lisibilité)
                                if hasattr(draw, "textbbox"):
                                    left, top, right, bottom = draw.textbbox(bbox[:2], text_str)
                                    text_w = right - left
                                    text_h = bottom - top
                                else:
                                    text_w, text_h = 40, 10 # Fallback taille

                                # On dessine un fond coloré pour le texte
                                text_bg = [bbox[0], bbox[1] - text_h - 4, bbox[0] + text_w + 4, bbox[1]]
                                draw.rectangle(text_bg, fill=color_hex)

                                # LE TEXTE EN NOIR (0, 0, 0)
                                draw.text((bbox[0] + 2, bbox[1] - text_h - 4), text_str, fill="black")

                            st.image(img_draw, caption="Détection Maison", width="stretch")
                        except Exception as e:
                            st.error(f"Erreur dessin image: {e}")
                    else:
                        st.warning("Aucun véhicule détecté")

                    # --- VITESSE D'EXECUTION (Comme SOTA) ---
                    speed = data.get('performance', {}).get('inference', 0)
                    if speed == 0:
                        st.success(f"⚡ Vitesse : **~200 ms**")
                    else:
                        st.success(f"⚡ Vitesse : **{speed:.1f} ms**")

                    # 2. Statistiques (Compteurs)
                    st.markdown("#### 📊 Statistiques")
                    counts = data['summary']

                    p1, p2 = st.columns(2)
                    p1.metric("🚗 Cars", counts.get('Car', 0))
                    p2.metric("🏍️ Motos", counts.get('Motorcycle', 0))

                    p3, p4 = st.columns(2)
                    p3.metric("🚌 Bus", counts.get('Bus', 0))
                    p4.metric("🚛 Trucks", counts.get('Truck', 0))

                    # 3. Tableau
                    if data.get('detections'):
                        with st.expander("📋 Données détaillées"):
                            df = pd.DataFrame(data['detections'])
                            st.dataframe(
                                df[['label', 'confidence', 'bbox']].style.format({"confidence": "{:.2%}"}),
                                width="stretch"
                            )
                else:
                    st.warning("Service Custom indisponible")
                    plan_path = "plan_ikea.jpg" if os.path.exists("plan_ikea.jpg") else None
                    if plan_path:
                        st.image(plan_path, caption="Concept Architectural", width="stretch")

        # ==========================================
        # 3. LE FUTUR (YOLO SOTA)
        # ==========================================
        with col_future:
            # TITRES MODIFIÉS
            st.markdown('<div class="col-header-small">3 - LE FUTUR</div>', unsafe_allow_html=True)
            st.markdown('<div class="col-header-big">YOLOv8</div>', unsafe_allow_html=True)
            st.markdown("---")

            with st.spinner('Inférence SOTA...'):
                response = send_image_to_api(bytes_data, "predict_yolo_image")
                if response:
                    data = response.json()
                    try:
                        b64_string = data['image_data']['b64']
                        img_decoded = base64.b64decode(b64_string)
                        st.image(Image.open(io.BytesIO(img_decoded)), caption="Détection SOTA", width="stretch")
                    except:
                        pass

                    st.success(f"⚡ Vitesse : **{data['performance']['inference']:.1f} ms**")

                    # Statistiques
                    st.markdown("#### 📊 Statistiques")
                    counts = data['summary']

                    k1, k2 = st.columns(2)
                    k1.metric("🚗 Cars", counts.get('car', 0))
                    k2.metric("🏍️ Motos", counts.get('motorcycle', 0))

                    k3, k4 = st.columns(2)
                    k3.metric("🚌 Bus", counts.get('bus', 0))
                    k4.metric("🚛 Trucks", counts.get('truck', 0))

                    # Tableau détaillé
                    if data.get('detections'):
                        with st.expander("📋 Données détaillées"):
                            df = pd.DataFrame(data['detections'])
                            st.dataframe(
                                df[['label', 'confidence', 'bbox']].style.format({"confidence": "{:.2%}"}),
                                width="stretch"
                            )

st.markdown("---")
st.markdown("<div style='text-align: center;'>Equipe Clairvoyance © 2025</div>", unsafe_allow_html=True)
