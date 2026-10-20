# Atelier 2 — Classification supervisée : Voiture / Bateau

Site pédagogique de classification supervisée d'images, basé sur l'atelier 2.

## Publication GRATUITE sur Streamlit Community Cloud

1. Ouvrir `entrainer_pour_site.ipynb` dans Google Colab.
2. Garder les images du dataset dans Google Drive : `Mon Drive/DB2C/obj_car/` et `Mon Drive/DB2C/obj_ship/`.
3. Exécuter les cellules d'entraînement puis la cellule **Exporter pour le site**.
4. Télécharger `export_site.zip`, et en extraire `modele_atelier.joblib` et `requirements.txt`.
5. Créer un **dépôt public** sur GitHub (bouton **New repository**).
6. Placer dans ce dépôt les fichiers `streamlit_app.py`, `caracteristiques.py`, `requirements.txt` (celui exporté de Colab), `modele_atelier.joblib` et `README.md`.
7. Aller sur https://share.streamlit.io, se connecter avec GitHub, sélectionner **Create app**.
8. Choisir ce dépôt, la branche `main`, puis `streamlit_app.py` et cliquer **Deploy**.
9. Copier le lien public `https://...streamlit.app/` pour le professeur.

**Important :** le lien ne fonctionne avec les prédictions que si `modele_atelier.joblib` a été produit à partir du VRAI dataset. Le ZIP fourni ne contient pas ce modèle car le dataset DB2C n'a pas été fourni ici.

## Tester en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Que fait le site ?

- Charger une photo librement (JPG/PNG/WebP, etc.).
- Classifier **Voiture / Bateau** avec les **six** modèles de l'atelier.
- Afficher la prédiction SVM RBF, celle du meilleur modèle d'après le test et toutes les prédictions.
- Afficher la matrice de confusion, les métriques et le classement obtenu à l'entraînement.
- Montrer les 52 caractéristiques calculées pour la photo téléchargée.

**Limite** : une image d'une classe inconnue sera obligatoirement étiquetée Voiture ou Bateau. La performance sur le dataset de test n'est pas la confiance pour une photo ajoutée.

**Sécurité** : le modèle joblib est un fichier sérialisé Python. Chargez uniquement votre propre export, jamais un joblib inconnu.
