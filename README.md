# Déploiement de modèles Deep Learning avec Flask

Application web Flask servant deux modèles de deep learning :

| Modèle | Tâche | Architecture | Sortie |
|--------|-------|--------------|--------|
| `resnet18_types.pth` | Classification d'images | ResNet18 (fine-tuné) | 3 classes |
| `unet_best.pth` | Segmentation binaire | U-Net | Masque 1 canal |

## Structure du projet

```
.
├── app.py                  # Application Flask (routes + API)
├── models.py               # Architectures U-Net/ResNet18 + preprocessing + inférence
├── templates/
│   └── index.html          # Interface web (upload + résultats)
├── requirements.txt        # Dépendances Python
├── resnet18_types.pth      # Poids ResNet18
└── unet_best.pth           # Poids U-Net
```

## Installation

```bash
pip install -r requirements.txt
```

> PyTorch (torch + torchvision) nécessite une installation compatible avec votre
> configuration CUDA. Voir [pytorch.org](https://pytorch.org/get-started/locally/)
> si vous utilisez le GPU.

## Lancement

```bash
python app.py
```

Puis ouvrez **http://127.0.0.1:5000** dans votre navigateur.

## Interface web

1. Choisissez la tâche (Classification ou Segmentation)
2. Importez une image (glisser-déposer ou clic)
3. Cliquez sur « Lancer la prédiction »
4. Visualisez le résultat :
   - **Classification** : classe prédite, confiance, barres de probabilités
   - **Segmentation** : masque binaire (zones segmentées en blanc) + ratio de surface

## API REST

### Santé du service
```
GET /health
```

### Classification
```
POST /predict/classify
Content-Type: multipart/form-data
Body: image=<fichier image>
```
Réponse :
```json
{
  "success": true,
  "result": {
    "predicted_class": 1,
    "predicted_label": "Type 1",
    "confidence": 0.873,
    "probabilities": { "Type 0": 0.05, "Type 1": 0.87, "Type 2": 0.08 }
  }
}
```

### Segmentation
```
POST /predict/segment
Content-Type: multipart/form-data
Body: image=<fichier image>
```
Réponse :
```json
{
  "success": true,
  "result": {
    "mask": "data:image/png;base64,<...>",
    "foreground_ratio": 0.342
  }
}
```

## Personnalisation

### Noms des classes (ResNet18)
Les noms des 3 classes sont définis dans `models.py` :

```python
CLASS_NAMES = ["Type 0", "Type 1", "Type 2"]
```

Remplacez-les par les vraies classes de votre jeu d'entraînement.

### Tailles d'entrée
```python
RESNET_INPUT_SIZE = 224   # Classification
UNET_INPUT_SIZE   = 256   # Segmentation
```

### Normalisation
Les deux modèles utilisent la normalisation ImageNet (mean/std). Si vos modèles
ont été entraînés avec une normalisation différente, ajustez `IMAGENET_MEAN` et
`IMAGENET_STD` dans `models.py`.

## Exemple d'appel API avec cURL
```bash
curl -X POST http://127.0.0.1:5000/predict/classify \
     -F "image=@photo.jpg"
```
