# Prédiction des Passages à Niveau Dangereux en Tunisie
##  Etudiante : 
Rouatbi Hedil 

##  Description du projet : 
Ce projet utilise le Machine Learning pour prédire si un passage à niveau en Tunisie est dangereux ou non.

L’objectif est d’aider à améliorer la sécurité ferroviaire en analysant les caractéristiques des passages à niveau.

##  Problématique

Peut-on prédire le caractère dangereux d’un passage à niveau à partir de ses données géographiques et structurelles ?

##  Dataset

- Fichier : `dataset_final.xlsx`
- Données sur les passages à niveau en Tunisie
- Variables :
  - Zone
  - Gouvernorat
  - Nombre d’intersections
  - Niveau de sécurité
  - Mois
  - Latitude / Longitude
  - Nombre de victimes


##  Pipeline Machine Learning

1. Chargement des données
2. Nettoyage et preprocessing
3. Encodage des variables
4. Split Train / Test
5. Entraînement des modèles
6. Validation croisée
7. Comparaison des performances

## Modèles utilisés

- Random Forest
- XGBoost
- Gradient Boosting
- SVM
- KNN

##  Résultats

| Modèle            | Accuracy |
|------------------|----------|
| Random Forest    | 0.95    |
| XGBoost          | 0.97     |
| Gradient Boost   | 0.98     |
| SVM              | 0.90     |
| KNN              | 0.82     |


##  Application Streamlit

Une application interactive permet :

- Prédiction en temps réel
- Visualisation des zones dangereuses
- Carte interactive
- Bouton RUN pour tester les modèles

### Lancer l'application :

```bash
streamlit run streamlit/app.py