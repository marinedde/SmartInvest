# COMMENTAIRES

Tentative de regrouper les lignes en quelques catégories plutôt qu’en arrondissements.
L’encodage OneHot des arrondissements génère une vingtaine de colonnes. Nous avons donc décidé de réduire ce nombre de catégories (arrondissements) en appliquant du clustering.

## Observations 

### Kmeans

En utilisant K-Means sur les coordonnées latitude et longitude, nous parvenons facilement à regrouper les données en 5 à 6 catégories.

### DBSCAN

Avec DBSCAN, même en faisant varier eps et min_samples dans tous les sens, nous n’arrivons pas à obtenir un nombre réduit de catégories.
Souvent, on se retrouve avec plus de 60 clusters et, en observant la distribution, un seul cluster contient environ 90 % des appartements.

## Conclusion 

Dans notre cas, où l’objectif est de réduire le nombre de catégories avant de faire un OneHotEncoder sur les arrondissements, K-Means semble plus adapté que DBSCAN.

