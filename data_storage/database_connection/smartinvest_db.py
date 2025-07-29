from dotenv import load_dotenv
import mysql.connector as sql
import os

class Connection:
    def __init__(self):
        load_dotenv()
        self.conn = sql.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME")
        )
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_data (id SERIAL PRIMARY KEY,
                                            surface_reelle_bati FLOAT,
                                            prix_m2 FLOAT,
                                            nombre_pieces_principales FLOAT,
                                            longitude FLOAT,
                                            latitude FLOAT,
                                            nombre_lots INTEGER,
                                            annee INTEGER,
                                            arrondissement INTEGER,
                                            distance_datashop_km FLOAT,
                                            distance_espace_vert_km FLOAT,
                                            distance_college_km FLOAT,
                                            distance_universite_km FLOAT,
                                            distance_ecole_km FLOAT,
                                            distance_metro_km FLOAT,
                                            distance_TER_km FLOAT,
                                            distance_POI_min_km FLOAT,
                                            proche_POI_1km INTEGER,
                                            nb_POIs_inf_1km INTEGER,
                                            cle_interop_adr_proche TEXT,
                                            distance_batiment_m FLOAT,
                                            annee_construction_dpe FLOAT,
                                            zone INTEGER)""")
        self.conn.commit()
        print("Connexion parfaitement établie")
        
    def insert_data(self, surface_reelle_bati,prix_m2,nombre_pieces_principales,longitude,latitude,nombre_lots,annee,arrondissement,distance_datashop_km,distance_espace_vert_km,distance_college_km,distance_universite_km,distance_ecole_km,distance_metro_km,distance_TER_km,distance_POI_min_km,proche_POI_1km,nb_POIs_inf_1km,cle_interop_adr_proche,distance_batiment_m,annee_construction_dpe,zone):
        requete_SQL = """INSERT INTO raw_data (surface_reelle_bati,
                                            prix_m2,
                                            nombre_pieces_principales,
                                            longitude,
                                            latitude,
                                            nombre_lots,
                                            annee,
                                            arrondissement,
                                            distance_datashop_km,
                                            distance_espace_vert_km,
                                            distance_college_km,
                                            distance_universite_km,
                                            distance_ecole_km,
                                            distance_metro_km,
                                            distance_TER_km,
                                            distance_POI_min_km,
                                            proche_POI_1km,
                                            nb_POIs_inf_1km,
                                            cle_interop_adr_proche,
                                            distance_batiment_m,
                                            annee_construction_dpe,
                                            zone) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        values = (surface_reelle_bati,prix_m2,nombre_pieces_principales,longitude,latitude,nombre_lots,annee,arrondissement,distance_datashop_km,distance_espace_vert_km,distance_college_km,distance_universite_km,distance_ecole_km,distance_metro_km,distance_TER_km,distance_POI_min_km,proche_POI_1km,nb_POIs_inf_1km,cle_interop_adr_proche,distance_batiment_m,annee_construction_dpe,zone)
        self.cursor.execute(requete_SQL, values)
        self.conn.commit()
        print("insertion réussie")
        
        
    def updateColonneZone(self, zone, id_ligne):
        requete_sql = """
        UPDATE raw_data SET zone = %s WHERE id = %s
        """
        values = (zone,id_ligne)
        self.cursor.execute(requete_sql, values)
        self.conn.commit()
        print("update réussie")

    def sayHello():
        print('hello World')

    def close_connection(self):
        self.cursor.close()
        self.conn.close()