import pandas as pd
from datetime import datetime


def filter_dataset(df):
    print("==== Filtrage du dataset ====")
    print(f"Nombre initial de lignes: {len(df)}")
    # Créer une copie pour éviter les avertissements
    df = df.copy()
    len_initial = len(df)
    # Vérifier les données vides - garder seulement les lignes avec latitude, longitude et date
    df = df.dropna(subset=['latitude', 'longitude', 'date'])
    print(f"Après suppression des données vides: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)

    # Supprimer les doublons
    df = df.drop_duplicates()
    print(f"Après suppression des doublons: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)

    # Supprimer les doublons de coordonnées
    df = df.drop_duplicates(subset=['latitude', 'longitude'])
    print(f"Après suppression des doublons de coordonnées: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)

    # Vérifier le format de la date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    print(f"Après vérification du format de la date: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)
    
    # Vérifier les coordonnées hors de Lyon
    lyon_coords = (45.7640, 4.8357)
    df = df[(df['latitude'].between(lyon_coords[0] - 0.1, lyon_coords[0] + 0.1)) &
              (df['longitude'].between(lyon_coords[1] - 0.1, lyon_coords[1] + 0.1))]
    print(f"Après filtrage des coordonnées hors de Lyon: {len_initial-len(df)} lignes filtrées.")
    return df

def load_and_prepare_data():
    # Charger le CSV
    df = pd.read_csv('flickr_data2.csv', low_memory=False)

    # Supprimer les espaces au début des noms de colonnes
    df.columns = df.columns.str.strip()

    # Nettoyer les colonnes de date - supprimer les caractères spéciaux (comme ";-)")
    date_columns = ['date_taken_year', 'date_taken_month', 'date_taken_day', 'date_taken_hour']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^0-9]', '', regex=True)

    # Renommer les colonnes pour correspondre à ce que filter_dataset attend
    df = df.rename(columns={'lat': 'latitude', 'long': 'longitude'})

    # Créer une colonne 'date' à partir des colonnes de date individuelles
    df['date'] = pd.to_datetime(
        df[['date_taken_year', 'date_taken_month', 'date_taken_day']].rename(
            columns={'date_taken_year': 'year', 'date_taken_month': 'month', 'date_taken_day': 'day'}
        ),
        errors='coerce'
    )
    
    # Process the hour column
    if 'date_taken_hour' in df.columns:
        df['hour'] = pd.to_numeric(df['date_taken_hour'], errors='coerce')
    else:
        df['hour'] = float('nan')
        
    return df


# Convertir en dictionnaire
def convert_to_dict_filtered():
    df=load_and_prepare_data()
    df_filtered = filter_dataset(df)
    print(f"Nombre de lignes conservées: {len(df_filtered)}")
    print(f"Nombre de lignes supprimées: {len(df) - len(df_filtered)} \n")

    return df_filtered

if __name__ == "__main__":
    convert_to_dict_filtered()
