from data_storage.database_connection.smartinvest_db import Connection
import pandas as pd

def exploData(path):
    """
        Fonction pour explorer le fichier .csv avant d'intégrer les lignes en bdd
    """
    with open(f'{path}','r') as f:
        lines = f.readlines()
        for index, i in enumerate(lines):
            line = i.split(',')
            if line[1] == 'surface_reelle_bati':
                continue
            print(line)
            if index == 2 :
                break


def integration():
    cn = Connection()
    with open('./data/data_final_SmartInvest.csv') as f:
        lines = f.readlines()
        for index, i in enumerate(lines) :
            line = i.split(',')
            # Titres des colonnes à ne pas insérer en bdd
            if line[1] == 'surface_reelle_bati':
                continue
            # attribution des lignes
            surface_reelle_bati = line[1]
            nombre_pieces_principales = line[2]
            longitude = line[3]
            latitude = line[4]
            nombre_lots = line[5]
            prix_m2 = line[6]
            annee = line[7]
            arrondissement = line[8]
            distance_datashop_km = line[9]
            distance_espace_vert_km = line[10]
            distance_college_km = line[11]
            distance_universite_km = line[12]
            distance_ecole_km = line[13]
            distance_metro_km = line[14]
            distance_TER_km = line[15]
            distance_POI_min_km = line[16]
            proche_POI_1km = line[17]
            nb_POIs_inf_1km = line[18]
            cle_interop_adr_proche = line[19]
            distance_batiment_m = line[20]
            annee_construction_dpe = line[21].replace('\n','')

            cn.insert_data(surface_reelle_bati,prix_m2,nombre_pieces_principales,longitude,latitude,nombre_lots,annee,arrondissement,distance_datashop_km,distance_espace_vert_km,distance_college_km,distance_universite_km,distance_ecole_km,distance_metro_km,distance_TER_km,distance_POI_min_km,proche_POI_1km,nb_POIs_inf_1km,cle_interop_adr_proche,distance_batiment_m,annee_construction_dpe)
            print(f"ligne {index} bien enregistré")
    cn.close_connection()
        