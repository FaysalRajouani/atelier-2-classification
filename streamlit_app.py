# -*- coding: utf-8 -*-
"""Application Streamlit — Atelier 2 (voiture / bateau, six classifieurs).

Nécessite le fichier modele_atelier.joblib exporté depuis
entrainer_pour_site.ipynb sur les VRAIES données DB2C.
"""
from pathlib import Path
import io
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from caracteristiques import adapter_taille, extract_one

st.set_page_config(
    page_title="Atelier 2 · Classification d'images",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1120px; padding-top: 2rem;}
.mini-label {color: #6b7280; font-size: .89rem;}
.result {padding: 1rem 1.1rem; background: #ecfdf5; border: 1px solid #86efac;
    border-radius: .8rem; color: #14532d; margin: .6rem 0;}
</style>
""", unsafe_allow_html=True)
st.title("🔎 Atelier 2 — Classification supervisée d'images")
st.caption("RGB · HSV · GLCM · LBP · moments de Hu · SVM · KNN · arbre de décision")

CHEMIN_MODELE = Path(__file__).resolve().parent / "modele_atelier.joblib"

@st.cache_resource(show_spinner=False)
def charger_artefact():
    # Ne charger QUE le fichier joblib produit par votre propre Colab.
    return joblib.load(CHEMIN_MODELE)

if not CHEMIN_MODELE.is_file():
    st.warning("Le site est prêt, mais le modèle entraîné manque encore.")
    st.markdown("""
    Pour activer les prédictions, ouvrez **entrainer_pour_site.ipynb** dans Google Colab,
    entraînez les six modèles sur votre dossier **DB2C** (obj_car et obj_ship),
    puis ajoutez **modele_atelier.joblib** à ce dépôt GitHub.
    Le téléchargement est inclus dans le notebook.
    """)
    st.stop()

try:
    artefact = charger_artefact()
except Exception as erreur:
    st.error("Impossible de charger le modèle. Vérifiez la compatibilité des versions dans requirements.txt.")
    st.exception(erreur)
    st.stop()

modeles = artefact["modeles"]
classes = {int(k): v for k, v in artefact["classes"].items()}
taille = tuple(artefact["taille_reference"])
evaluations = artefact["evaluations"]
classement = sorted(evaluations, key=lambda nom: evaluations[nom]["accuracy"], reverse=True)
meilleur_modele = classement[0]

onglet_image, onglet_resultats, onglet_methode = st.tabs([
    "📤 Tester une image", "📊 Résultats de l'atelier", "🧪 Méthode"
])

with onglet_image:
    st.subheader("Choisissez une image depuis votre appareil")
    st.write("Importez **une nouvelle photo**, puis cliquez sur **Classer cette image**. Vous pouvez recommencer autant de fois que vous voulez sans réentraîner les modèles.")
    televersement = st.file_uploader(
        "Image à classifier", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=False,
    )
    vraie_classe = st.selectbox("Classe réelle (facultatif, si elle est connue)",
                                ["Inconnue", "Voiture", "Bateau"])
    if televersement is not None:
        contenu = televersement.getvalue()
        matrice = cv2.imdecode(np.frombuffer(contenu, dtype=np.uint8), cv2.IMREAD_COLOR)
        if matrice is None:
            st.error("Image illisible. Essayez un fichier JPG ou PNG valide.")
        else:
            photo_rgb = cv2.cvtColor(matrice, cv2.COLOR_BGR2RGB)
            gauche, droite = st.columns([1, 1.25], gap="large")
            with gauche:
                st.image(photo_rgb, caption=televersement.name, use_container_width=True)
                st.caption(f"Dimensions de l'image : {photo_rgb.shape[1]} × {photo_rgb.shape[0]} pixels")
            with droite:
                if st.button("🔍 Classer cette image", type="primary", use_container_width=True):
                    try:
                        image_preparee = adapter_taille(photo_rgb, taille)
                        vecteur = extract_one(image_preparee).reshape(1, -1).astype(np.float32)
                        assert vecteur.shape == (1, 52)
                        predictions = {
                            nom: classes[int(modele.predict(vecteur)[0])]
                            for nom, modele in modeles.items()
                        }
                        resultat_svm = predictions["SVM RBF"]
                        st.markdown(f'<div class="result"><b>Prédiction SVM RBF : {resultat_svm}</b></div>',
                                    unsafe_allow_html=True)
                        st.write(f"**Meilleur algorithme sur le jeu de test :** {meilleur_modele} → **{predictions[meilleur_modele]}**")
                        if vraie_classe != "Inconnue":
                            msg = "Prédiction correcte ✔️" if resultat_svm == vraie_classe else "Prédiction incorrecte ✖️"
                            st.write(f"**Vraie classe indiquée :** {vraie_classe} — {msg} (SVM RBF)")
                        tableau = pd.DataFrame([
                            {"Algorithme": nom, "Prédiction": predictions[nom],
                             "Accuracy sur le test": f"{evaluations[nom]['accuracy'] * 100:.2f} %"}
                            for nom in classement
                        ])
                        st.markdown("**Les prédictions des six algorithmes :**")
                        st.dataframe(tableau, use_container_width=True, hide_index=True)
                        st.info("Les accuracies indiquées ont été mesurées sur le jeu de test du dataset, PAS sur cette photo. Ce ne sont pas des probabilités de confiance.")
                        with st.expander("Détails techniques : 52 caractéristiques de la nouvelle image"):
                            st.write(f"Image redimensionnée à **{taille[0]} × {taille[1]} pixels** (mêmes dimensions qu'à l'entraînement).")
                            st.dataframe(pd.DataFrame({
                                "N°": range(1, 53),
                                "Caractéristique": (["RGB"] * 6 + ["HSV"] * 24 +
                                                     ["GLCM"] * 4 + ["LBP"] * 11 + ["Hu"] * 7),
                                "Valeur": vecteur.ravel(),
                            }), use_container_width=True, hide_index=True)
                    except Exception as erreur:
                        st.error("Échec de la classification.")
                        st.exception(erreur)
    else:
        st.info("Aucune image choisie. Le professeur peut importer une photo ici pour tester le modèle.")
    st.warning("⚠️ Le modèle ne connaît que deux classes : **Voiture** et **Bateau**. Il donnera forcément l'une de ces classes même si l'image contient un autre objet.")

with onglet_resultats:
    st.subheader("Évaluation sur le jeu de test (20 % du dataset DB2C)")
    st.caption("Résultats réels calculés lors de l'entraînement exporté, sans réentraîner à chaque visite.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Images d'entraînement", artefact.get("nb_train", "—"))
    col2.metric("Images de test", artefact.get("nb_test", "—"))
    col3.metric("Accuracy SVM RBF", f"{evaluations['SVM RBF']['accuracy'] * 100:.2f} %")
    st.markdown("#### Comparaison des six algorithmes")
    valeurs = pd.DataFrame({
        "Algorithme": classement,
        "Accuracy (%)": [round(evaluations[n]["accuracy"] * 100, 2) for n in classement],
    })
    st.bar_chart(valeurs.set_index("Algorithme"), y="Accuracy (%)", horizontal=True, height=330)
    st.dataframe(valeurs, hide_index=True, use_container_width=True)
    st.success(f"Meilleur modèle sur le jeu de test : {meilleur_modele} ({evaluations[meilleur_modele]['accuracy'] * 100:.2f} %)")
    st.markdown("#### Matrice de confusion — SVM RBF")
    cm = np.array(evaluations["SVM RBF"]["confusion_matrix"], dtype=int)
    fig, ax = plt.subplots(figsize=(5, 4))
    rendu = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=[classes[0], classes[1]])
    ax.set_yticks([0, 1], labels=[classes[0], classes[1]])
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")
    for (ligne, colonne), nombre in np.ndenumerate(cm):
        ax.text(colonne, ligne, str(nombre), ha="center", va="center",
                color="white" if nombre > (cm.max() * .5) else "black")
    fig.colorbar(rendu, ax=ax, shrink=.8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.markdown("#### Rapport de classification — SVM RBF")
    rapport = evaluations["SVM RBF"]["classification_report"]
    lignes = []
    for nom in [classes[0], classes[1], "macro avg", "weighted avg"]:
        if nom in rapport:
            entry = rapport[nom]
            lignes.append({"Classe": nom, "Précision": round(entry["precision"], 4),
                           "Rappel": round(entry["recall"], 4),
                           "F1-score": round(entry["f1-score"], 4), "Support": int(entry["support"])})
    st.dataframe(pd.DataFrame(lignes), use_container_width=True, hide_index=True)
    st.caption("Précision, rappel, F1-score et support concernent le jeu de test étiqueté, et non une image librement importée.")

with onglet_methode:
    st.subheader("Chaîne de traitement de l'atelier")
    st.markdown("""
    **1. Jeu de données** — dossiers DB2C/obj_car (Voiture) et DB2C/obj_ship (Bateau). 
    Séparation apprentissage / test : 80 % / 20 % avec `random_state=42`, stratification.

    **2. Prétraitement** — lecture couleur RGB, mise à la résolution utilisée à l'entraînement.

    **3. Extraction de 52 caractéristiques** — 6 statistiques RGB, 24 valeurs d'histogrammes HSV,
    4 propriétés GLCM, 11 valeurs LBP et 7 moments de Hu.

    **4. Classification** — SVM RBF (principal), SVM linéaire, SVM sigmoïde, KNN (3),
    KNN (5), arbre de décision. StandardScaler est utilisé pour SVM et KNN.

    **5. Évaluation** — matrice de confusion, accuracy, précision, rappel, F1-score sur les images
    **réservées au test**, puis comparaison des six modèles.

    **6. Nouvelle image** — calcul des mêmes 52 caractéristiques et prédiction à partir des modèles déjà entraînés.
    """)
    st.caption("Travail pratique : classification supervisée, démonstration pédagogique.")
