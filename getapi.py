import requests
import datetime
import pandas as pd

# 1️⃣ URL de l'API open data Rennes Métropole
url = "https://data.rennesmetropole.fr/api/records/1.0/search/"

# 2️⃣ Paramètres de la requête
params = {
    "dataset": "etat-du-trafic-en-temps-reel",
    "rows": 100,         # nombre d’enregistrements
    "sort": "-datetime"  # trier du plus récent au plus ancien
}

# 3️⃣ Appel de l’API
response = requests.get(url, params=params)
response.raise_for_status()  # pour détecter les erreurs HTTP
data = response.json()

# 4️⃣ Extraction des champs pertinents dans un tableau
records = []
for rec in data.get("records", []):
    fields = rec.get("fields", {})
    records.append({
        "Date/Heure": fields.get("datetime"),
        "Route": fields.get("denomination"),
        "Vitesse Moyenne (km/h)": fields.get("averagevehiclespeed"),
        "Vitesse Max (km/h)": fields.get("vitesse_maxi"),
        "Temps de Trajet (s)": fields.get("traveltime"),
        "Fiabilité (%)": fields.get("traveltimereliability"),
        "Statut Trafic": fields.get("trafficstatus"),
        "Hiérarchie": fields.get("hierarchie"),
    })

# 5️⃣ Chargement dans un DataFrame Pandas
df = pd.DataFrame(records)

# 6️⃣ Affichage tabulaire
print("\n=== Aperçu des données de trafic à Rennes ===\n")
print(df.head(10))  # affiche les 10 premières lignes