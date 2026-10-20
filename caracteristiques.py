# -*- coding: utf-8 -*-
"""Extraction strictement conforme à l'atelier 2 : 52 caractéristiques.

RGB(6) + HSV(24) + GLCM(4) + LBP(11) + moments de Hu(7).
Le modèle attend une image RGB (numpy H x L x 3, uint8).
"""
import cv2
import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


def adapter_taille(image, taille_reference):
    largeur, hauteur = tuple(int(x) for x in taille_reference)
    if image.shape[:2] == (hauteur, largeur):
        return image
    interpolation = (cv2.INTER_AREA if
                     (image.shape[1] > largeur or image.shape[0] > hauteur)
                     else cv2.INTER_LINEAR)
    return cv2.resize(image, (largeur, hauteur), interpolation=interpolation)


def extract_one(image):
    rgb = image.astype(np.float32) / 255.0
    rgb_stats = np.concatenate((
        rgb.mean(axis=(0, 1)),
        rgb.std(axis=(0, 1))
    ))
    hsv = rgb2hsv(rgb)
    hsv_hist = np.concatenate([
        np.histogram(hsv[:, :, channel], bins=8, range=(0, 1))[0]
        / (rgb.shape[0] * rgb.shape[1])
        for channel in range(3)
    ])
    gray8 = np.round(rgb2gray(rgb) * 255).astype(np.uint8)
    quantized = (gray8 // 32).astype(np.uint8)
    glcm = graycomatrix(
        quantized,
        distances=[1, 2],
        angles=[0, np.pi / 2],
        levels=8,
        symmetric=True,
        normed=True,
    )
    glcm_features = np.array([
        graycoprops(glcm, prop).mean()
        for prop in ('contrast', 'dissimilarity', 'homogeneity', 'energy')
    ])
    lbp = local_binary_pattern(gray8, P=8, R=1, method='uniform')
    lbp_hist = np.histogram(lbp, bins=np.arange(0, 12))[0].astype(np.float64)
    lbp_hist /= max(lbp_hist.sum(), 1)
    hu = cv2.HuMoments(cv2.moments(gray8)).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    features = np.concatenate((rgb_stats, hsv_hist, glcm_features, lbp_hist, hu))
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
