"""
Application Flask de déploiement des modèles de deep learning.

Endpoints
---------
GET  /                      Page d'accueil (interface web)
POST /predict/classify      Classification d'image (ResNet18, 3 classes)
POST /predict/segment       Segmentation binaire (U-Net)
GET  /health                État du service + modèles chargés

Lancement
---------
    python app.py
    # puis ouvrir http://127.0.0.1:5000
"""
from __future__ import annotations

import base64
import io
import logging

from flask import Flask, jsonify, render_template, request

import models
from models import (
    CLASS_NAMES,
    load_resnet,
    load_unet,
    predict_classification,
    predict_segmentation,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration & logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("flask-deploy")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 Mo max par upload

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp", "tiff"}


# ---------------------------------------------------------------------------
# Chargement des modèles au démarrage (une seule fois)
# ---------------------------------------------------------------------------
logger.info("Chargement des modèles...")
resnet_model = load_resnet()
unet_model = load_unet()
logger.info("Modèles chargés: ResNet18 (%d classes) + U-Net (segmentation binaire)",
            len(CLASS_NAMES))


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def read_uploaded_image() -> Image.Image:
    """Lit l'image envoyée dans le champ 'image' de la requête.

    Lève une ValueError si le fichier est absent ou illisible.
    """
    if "image" not in request.files:
        raise ValueError("Aucun fichier envoyé (champ 'image' manquant).")

    file = request.files["image"]
    if file.filename == "":
        raise ValueError("Aucun fichier sélectionné.")

    if not allowed_file(file.filename):
        raise ValueError(
            f"Format non supporté. Formats acceptés: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Image illisible: {exc}") from exc

    return image


def mask_to_base64_png(mask_image: Image.Image) -> str:
    """Convertit un masque PIL en chaîne base64 (PNG data URI)."""
    buf = io.BytesIO()
    mask_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/demo")
def index():
    return render_template("index.html", classes=CLASS_NAMES)


@app.route("/health")
def health():
    return jsonify(
        status="ok",
        models={
            "resnet18": {"type": "classification", "num_classes": len(CLASS_NAMES)},
            "unet": {"type": "segmentation", "output_channels": 1},
        },
        device=str(models.DEVICE),
    )


@app.route("/predict/classify", methods=["POST"])
def classify():
    """Classification d'image avec ResNet18.

    Réponse JSON:
        {
            "success": true,
            "result": {
                "predicted_class": 1,
                "predicted_label": "Type 1",
                "confidence": 0.87,
                "probabilities": {"Type 0": 0.05, "Type 1": 0.87, "Type 2": 0.08}
            }
        }
    """
    try:
        image = read_uploaded_image()
        result = predict_classification(resnet_model, image)
        logger.info("Classification: %s (%.1f%%)", result["predicted_label"],
                    result["confidence"] * 100)
        return jsonify(success=True, result=result)
    except ValueError as exc:
        return jsonify(success=False, error=str(exc)), 400
    except Exception as exc:
        logger.exception("Erreur lors de la classification")
        return jsonify(success=False, error=f"Erreur interne: {exc}"), 500


@app.route("/predict/segment", methods=["POST"])
def segment():
    """Segmentation binaire avec U-Net.

    Réponse JSON:
        {
            "success": true,
            "result": {
                "mask": "<base64 PNG data URI>",
                "foreground_ratio": 0.34
            }
        }
    """
    try:
        image = read_uploaded_image()
        seg = predict_segmentation(unet_model, image)
        mask_b64 = mask_to_base64_png(seg["mask_image"])
        logger.info("Segmentation: %.1f%% premier plan", seg["foreground_ratio"] * 100)
        return jsonify(
            success=True,
            result={
                "mask": f"data:image/png;base64,{mask_b64}",
                "foreground_ratio": round(seg["foreground_ratio"], 4),
            },
        )
    except ValueError as exc:
        return jsonify(success=False, error=str(exc)), 400
    except Exception as exc:
        logger.exception("Erreur lors de la segmentation")
        return jsonify(success=False, error=f"Erreur interne: {exc}"), 500


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
