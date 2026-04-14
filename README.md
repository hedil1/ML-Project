# Prédiction des Passages à Niveau Dangereux en Tunisie
##  Etudiante : 
Rouatbi Hedil 

##  Description du projet : 
Ce projet utilise le Machine Learning pour prédire si un passage à niveau en Tunisie est dangereux ou non.

L’objectif est d’aider à améliorer la sécurité ferroviaire en analysant les caractéristiques des passages à niveau.

##  Problématique

Peut-on prédire le caractère dangereux d’un passage à niveau à partir de ses données géographiques et structurelles ?

##  Dataset

- **Fichier** : `dataset_final.xlsx`
- **Source** : Les accidents SNCFT 
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
5. Normalisation des données
6. Entraînement des modèles
7. Validation croisée (5 folds)
8. Comparaison des performances

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

**Meilleur modèle : Gradient Boosting (Accuracy 98.97%)**


##  Application Streamlit

Une application interactive permet :

- Prédiction en temps réel
- Visualisation des zones dangereuses
- Carte interactive
- Bouton RUN pour tester les modèles

##  Application Streamlit

https://ml-project-txgok9tgpaavaymrpjybw4.streamlit.app/



### Fonctionnalités de l'application

 **Prédiction en temps réel** - Testez n'importe quel passage à niveau  
 **Carte interactive** - Visualisez les zones dangereuses  
 **Heatmap dynamique** - Filtrage par mois  
 **Exploration par gouvernorat** - Détails par lieu avec liste des passages  
 **5 modèles ML** - Choisissez le modèle pour la prédiction  
 **Statistiques détaillées** - Taux de danger, nombre de victimes

 ### Lancer l'application localement

 ```bash
# Cloner le repository
git clone https://github.com/hedil1/ML-Project.git
cd ML-Project

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'application
streamlit run app.py