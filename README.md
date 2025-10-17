# Détection de Fuite de Produit avec YOLOv11

## Présentation
Ce projet propose une solution intelligente capable de détecter automatiquement les fuites de produit (liquide/eau) dans des images, des vidéos ou des flux de caméra en temps réel. Il exploite la puissance du modèle d’intelligence artificielle YOLOv11 pour l’analyse visuelle.

## Fonctionnalités Principales
- **Détection automatique des fuites:** dans les images, vidéos et webcam/flux IP
- **Interface web facile d’utilisation:** basée sur Streamlit
- **Téléchargement des résultats annotés**
- **Statistiques de détection**
- **Commenter le code en français pour une compréhension aisée**

## Prérequis
- Python 3.8+
- Bibliothèques : ultralytics, streamlit, opencv-python, cvzone, etc.
- Un modèle YOLOv11 entraîné (`best.pt` ou équivalent)

Installez les dépendances avec :
```bash
pip install -r requirements.txt
```

## Utilisation
1. **Lancer l’application web Streamlit :**
```bash
streamlit run app.py
```

2. **Choisir la source d’entrée** dans le menu de gauche :
    - Image (format jpg/png)
    - Vidéo (mp4, avi, mov…)
    - Webcam (ou caméra IP, à adapter selon l’environnement)

3. **Voir les résultats et télécharger les annotations**.

## Structure du projet
- `app.py` : interface principale (Streamlit)
- `detector.py` : classe de détection et fonctions utilitaires
- `fullcode.py` : exemple d’utilisation avancée avec tracking
- `best.pt` : modèle IA YOLOv11 (à ajouter)

## Conseils et Personnalisation
- Pour entraîner un modèle sur vos propres données, voir la documentation YOLO.
- Modifiez les commentaires en français pour mieux comprendre chaque partie du code.
- Extensible (multi-caméra, notifications, etc.)

## Auteurs
Développé par Ayoub Abdellaoui et contributeurs IA.

## Remerciements
Merci d’utiliser ce projet ! N’hésitez pas à l’adapter à vos besoins et à suggérer des améliorations.
