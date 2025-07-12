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
                                            valeur_fonciere FLOAT,
                                            surface_reelle_bati FLOAT,
                                            nombre_pieces_principales INTEGER,
                                            longitude FLOAT,
                                            latitude FLOAT,
                                            prix_m2 INTEGER ,
                                            jour TEXT,
                                            mois TEXT,
                                            annee TEXT,
                                            arrondissement TEXT)""")
        self.conn.commit()
        print("Connexion parfaitement établie")

    def insert_data(self, surface_reelle_bati,nombre_pieces_principales,longitude,latitude,prix_m2,jour,mois,annee,arrondissement):
        requete_SQL = """INSERT INTO raw_data (surface_reelle_bati,
                                        nombre_pieces_principales,
                                        longitude,
                                        latitude,
                                        prix_m2,
                                        jour,
                                        mois,
                                        annee,
                                        arrondissement) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)"""
        values = (surface_reelle_bati,nombre_pieces_principales,longitude,latitude,prix_m2,jour,mois,annee,arrondissement)
        self.cursor.execute(requete_SQL, values)
        self.conn.commit()
        print("insertion réussie")

    def close_connection(self):
        self.cursor.close()
        self.conn.close()