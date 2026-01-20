import pandas as pd
from datetime import datetime


def filter_dataset(df):
    print("==== Filtrage du dataset ====")
    print(f"Nombre initial de lignes: {len(df)}")
    # Créer une copie pour éviter les avertissements
    df = df.copy()
    len_initial = len(df)
    # Vérifier les données vides - garder seulement les lignes avec latitude,
    # longitude et date
    df = df.dropna(subset=['latitude', 'longitude', 'date'])
    print(
        f"Après suppression des données vides: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)

    # Supprimer les doublons
    df = df.drop_duplicates()
    print(
        f"Après suppression des doublons: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)

    # Vérifier le format de la date
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    print(
        f"Après vérification du format de la date: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)
    # Supprimer les dates futures
    current_date = datetime.now()
    df = df[df['date'] <= current_date]
    print(
        f"Après suppression des dates futures: {len_initial-len(df)} lignes filtrées.")
    len_initial = len(df)

    # Vérifier les coordonnées hors de Lyon
    haut_lyon_gauche_coords = (45.82, 4.752)
    bas_lyon_droite_coords = (45.67, 4.98)
    df = df[(df['latitude'].between(bas_lyon_droite_coords[0], haut_lyon_gauche_coords[0])) &
            (df['longitude'].between(haut_lyon_gauche_coords[1], bas_lyon_droite_coords[1]))]
    print(
        f"Après filtrage des coordonnées hors de Lyon: {len_initial-len(df)} lignes filtrées.")

    # Regrouper par coordonnées géographiques
    # Chaque point représente toutes les photos à cette localisation
    cols_to_agg = {
        'id': list,
        'title': list,
        'tags': list,
        'date': list,
        'user': list
    }
    # Only aggregate available columns
    agg_dict = {k: v for k, v in cols_to_agg.items() if k in df.columns}

    df_grouped = df.groupby(['latitude', 'longitude']
                            ).agg(agg_dict).reset_index()
    print(
        f"Après regroupement par coordonnées dans df_groupe: {len_initial} lignes regroupées en {len(df_grouped)} points géographiques.")

    # Vérifier doublons latitude/longitude
    df = df.drop_duplicates(subset=['latitude', 'longitude'])
    print(
        f"Après suppression des doublons géographiques: {len_initial-len(df)} lignes filtrées.")

    return df, df_grouped


def load_and_prepare_data():
    # Charger le CSV
    df = pd.read_csv('flickr_data2.csv', low_memory=False)

    # Supprimer les espaces au début des noms de colonnes
    df.columns = df.columns.str.strip()

    # Nettoyer les colonnes de date - supprimer les caractères spéciaux (comme
    # ";-)")
    date_columns = ['date_taken_year', 'date_taken_month',
                    'date_taken_day', 'date_taken_hour']
    for col in date_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                r'[^0-9]', '', regex=True)

    # Renommer les colonnes pour correspondre à ce que filter_dataset attend
    df = df.rename(columns={'lat': 'latitude', 'long': 'longitude'})

    # Créer une colonne 'date' à partir des colonnes de date individuelles
    df['date'] = pd.to_datetime(
        df[['date_taken_year', 'date_taken_month', 'date_taken_day']].rename(
            columns={'date_taken_year': 'year',
                     'date_taken_month': 'month', 'date_taken_day': 'day'}
        ),
        errors='coerce'
    )

    # Process the hour column
    if 'date_taken_hour' in df.columns:
        df['hour'] = pd.to_numeric(df['date_taken_hour'], errors='coerce')
    else:
        df['hour'] = float('nan')

    return df

# Get info by geo point


def get_info_geo(
        df: pd.DataFrame,
        latitude: float,
        longitude: float) -> dict | None:
    """Renvoie les informations d'un point géographique donné"""
    result = df[(df['latitude'] == latitude) & (df['longitude'] == longitude)]
    if not result.empty:
        return result.iloc[0].to_dict()
    return None

# Convertir en dictionnaire


def convert_to_dict_filtered():
    df = load_and_prepare_data()
    df_filtered = filter_dataset(df)
    print(f"Nombre de lignes conservées: {len(df_filtered[0])}")
    print(f"Nombre de lignes supprimées: {len(df) - len(df_filtered[0])} \n")

    return df_filtered[0]


if __name__ == "__main__":
    convert_to_dict_filtered()
