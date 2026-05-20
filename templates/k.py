# Sélection des colonnes binaires
polluants_binaires = [col for col in data_aqi.columns if col.endswith("_bin")]

# Création du tableau binaire pour Apriori
data_apriori = data_aqi[polluants_binaires]

#importation de la librairie  mlxtend

from mlxtend.frequent_patterns import apriori, association_rules

#Appliquer Apriori pour trouver les combinaisons fréquentes
# Fréquence minimale des combinaisons : 30% (modifiable)
frequent_itemsets = apriori(data_apriori, min_support=0.1, use_colnames=True)

print(frequent_itemsets)

#On définit comme pollutant influent celui qui apparaît dans les règles avec AQI élevé (ou présence binaire = 1).
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Filtrer les règles où le consequent contient un polluant binaire
rules = rules[rules['consequents'].apply(lambda x: any(item in polluants_binaires for item in x))]

# Afficher les règles triées par confiance
rules = rules.sort_values(by='confidence', ascending=False)
print(rules[['antecedents','consequents','support','confidence','lift']])
