from data_storage.sql_alchemy_connect.alchemy_cur import AlchemyConn
import pandas as pd


co = AlchemyConn()


df_from_db = pd.read_sql("SELECT * FROM raw_data", con=co.get_engine())
print(df_from_db.head())