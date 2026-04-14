# Prédiction des Passages à Niveau Dangereux en Tunisie
## Etudiante : 
Rouatbi Hedil 

##  Description du projet : 
La sécurité aux passages à niveau constitue un enjeu majeur de sécurité routière et ferroviaire en Tunisie. Chaque année, des collisions surviennent entre les trains et les véhicules à ces intersections critiques, causant des blessés et des victimes. La SNCFT (Société Nationale des Chemins de Fer Tunisiens) gère un réseau de passages à niveau répartis sur tout le territoire.

L'objectif de ce projet est de développer un modèle de Machine Learning capable d'identifier les passages à niveau présentant un risque élevé d'accident, afin d'orienter les efforts de maintenance et de sécurisation vers les points les plus critiques.

##  Problématique
Peut-on prédire le caractère dangereux d'un passage à niveau (classification binaire : dangereux / non dangereux) à partir de ses caractéristiques structurelles, géographiques et opérationnelles ?

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

##  Interprétation des résultats
- Les modèles basés sur les arbres (Gradient Boosting, XGBoost, Random Forest) sont les plus performants.
- Les modèles simples comme KNN et SVM montrent leurs limites sur ce problème.
- La différence de performance confirme que le dataset contient des relations non linéaires complexes.

**Meilleur modèle : Gradient Boosting (Accuracy 98.97%)**
Mais Puisque les scores sont tres eleves j'essayer de faire une croisée validation pour confirmer la fiablité des resultats

Afin de garantir la robustesse du modèle et éviter le surapprentissage, une validation croisée a été réalisée :
-Accuracy moyenne (CV) : 0.886
-Écart-type (Std) : 0.079

=>L’accuracy moyenne de 0.886 est inférieure aux résultats obtenus précédemment (jusqu’à 0.98) car les performances varient selon les sous-ensembles de donnees

=>L’écart-type de 0.079 montre une variabilité modérée, ce qui signifie que le modèle n’est pas totalement stable.
=>La validation croisée donne une estimation plus réaliste des performances.

**Le choix final dépend donc de l’objectif :**
1. Performance maximale → Gradient Boosting
2. Stabilité / robustesse → Random Forest

L’évaluation des modèles a été réalisée de manière rigoureuse en combinant :
1. test simple
2. validation croisée
3. comparaison de plusieurs algorithmes

Cette approche permet :

1. d’éviter les conclusions biaisées
2. sélectionner un modèle fiable
3.  comprendre les limites des performances obtenues

**Conclusion**
Même si certains modèles atteignent 98% en test, la validation croisée montre que la performance réelle est autour de 88%,

##  Application Streamlit
Une application interactive permet :

- Prédiction en temps réel
- Visualisation des zones dangereuses
- Carte interactive
- Bouton RUN pour tester les modèles

##  Application Streamli
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