"""
Définitions des modèles et fonctions de preprocessing.

- UNet      : segmentation binaire (1 canal de sortie, sigmoid)
- ResNet18  : classification d'images en 3 classes
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Constantes de preprocessing (normalisation ImageNet — standard pour les
# modèles pré-entraînés / fine-tunés sur ImageNet)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Taille d'entrée ResNet18 (standard torchvision)
RESNET_INPUT_SIZE = 224

# Taille d'entrée U-Net — 256x256 est un choix courant pour la segmentation ;
# les convolutions 3x3 + maxpool maintiennent des dimensions paires à chaque
# niveau (256 → 128 → 64 → 32 → 16), ce qui évite les problèmes d'arrondi.
UNET_INPUT_SIZE = 256

# Noms des classes pour le ResNet18.
# ⚠️  Adaptez ces noms aux vraies classes de votre jeu d'entraînement.
CLASS_NAMES = ["animals", "obstacles", "other"]

# Configuration du device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================================
#  U-Net
# ===========================================================================
class DoubleConv(nn.Module):
    """Bloc (Conv-BN-ReLU) × 2.  Les convolutions n'ont pas de biais car le
    BatchNorm qui suit l'absorbe."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(x)


class UNet(nn.Module):
    """U-Net pour segmentation binaire.

    Encodeur :  e1(3→64)  e2(64→128)  e3(128→256)  e4(256→512)
    Bottleneck : bt(512→1024)
    Décodeur :  d4(1024→512) d3(512→256) d2(256→128) d1(128→64)
    Tête     :  head(64→1)
    """

    def __init__(self):
        super().__init__()
        # --- Encodeur ---
        self.e1 = DoubleConv(3, 64)
        self.e2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.e3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.e4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- Bottleneck ---
        self.bt = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # --- Décodeur (up-conv + skip connection + DoubleConv) ---
        self.u4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d4 = DoubleConv(1024, 512)  # 512 (up) + 512 (skip e4)

        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d3 = DoubleConv(512, 256)  # 256 (up) + 256 (skip e3)

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d2 = DoubleConv(256, 128)  # 128 (up) + 128 (skip e2)

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d1 = DoubleConv(128, 64)  # 64 (up) + 64 (skip e1)

        # --- Tête 1×1 ---
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encodeur
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.bt(x4)

        # Décodeur + skip connections
        y = self.u4(x5)
        y = self.d4(torch.cat([y, x4], dim=1))

        y = self.u3(y)
        y = self.d3(torch.cat([y, x3], dim=1))

        y = self.u2(y)
        y = self.d2(torch.cat([y, x2], dim=1))

        y = self.u1(y)
        y = self.d1(torch.cat([y, x1], dim=1))

        return self.head(y)


# ===========================================================================
#  Loaders
# ===========================================================================
def load_unet(path: str = "../models/objectif_louay/unet_best.pth") -> UNet:
    """Charge l'U-Net et le met en mode évaluation."""
    model = UNet()
    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_resnet(path: str = "../models/objectif_louay/resnet18_types.pth") -> nn.Module:
    """Charge le ResNet18 fine-tuné pour 3 classes."""
    model = tv_models.resnet18(weights=None)
    model.fc = nn.Linear(512, len(CLASS_NAMES))
    state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


# ===========================================================================
#  Preprocessing
# ===========================================================================
def get_classification_transform() -> transforms.Compose:
    """Pipeline de preprocessing pour la classification (224×224)."""
    return transforms.Compose(
        [
            transforms.Resize((RESNET_INPUT_SIZE, RESNET_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_segmentation_transform() -> transforms.Compose:
    """Pipeline de preprocessing pour la segmentation (256×256)."""
    return transforms.Compose(
        [
            transforms.Resize((UNET_INPUT_SIZE, UNET_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ===========================================================================
#  Inférence
# ===========================================================================
@torch.inference_mode()
def predict_classification(
    model: nn.Module, image: Image.Image
) -> dict:
    """Exécute la classification et renvoie la classe + les probabilités."""
    transform = get_classification_transform()
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    idx = int(torch.argmax(probs))
    return {
        "predicted_class": idx,
        "predicted_label": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        },
    }


@torch.inference_mode()
def predict_segmentation(
    model: nn.Module, image: Image.Image
) -> dict:
    """Exécute la segmentation binaire et renvoie un masque (PNG base64).

    Le masque est redimensionné à la taille d'origine de l'image.
    """
    orig_w, orig_h = image.size
    transform = get_segmentation_transform()
    tensor = transform(image).unsqueeze(0).to(DEVICE)  # (1, 3, 256, 256)

    logits = model(tensor)  # (1, 1, 256, 256)
    prob = torch.sigmoid(logits).squeeze(0).squeeze(0)  # (256, 256)

    # Masque binaire (threshold 0.5) → image PIL en niveaux de gris
    mask = (prob > 0.5).byte().cpu()  # 0 / 1
    mask_img = Image.fromarray((mask * 255).numpy(), mode="L")
    mask_img = mask_img.resize((orig_w, orig_h), Image.NEAREST)

    # Statistiques rapides
    fg_ratio = float(mask.float().mean())

    return {
        "mask_image": mask_img,
        "foreground_ratio": fg_ratio,
    }
