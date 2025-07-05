from data_management.database_connection.smartinvest_db import Connection

cn = Connection()


with open('./data/model_db_all_years.csv') as f:
    lines = f.readlines()
    for i in lines :
        line = i.split(',')
        # attribution des lignes
        surface_reelle_bati = line[0]
        nombre_pieces_principales = line[1]
        longitude = line[2]
        latitude = line[3]
        prix_m2 = line[4]
        jour = line[5]
        mois = line[6]
        annee = line[7]
        # on enelève le suffix saut de ligne \n
        arrondissement = line[8].removesuffix("\n")

        cn.insert_data(surface_reelle_bati,nombre_pieces_principales,longitude,latitude,prix_m2,jour,mois,annee,arrondissement)

cn.close_connection()
        
