import os 
from dotenv import load_dotenv
from sqlalchemy import create_engine



class AlchemyConn : 
    def __init__(self):
        load_dotenv()
        DB_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        self.engine = create_engine(DB_URL)
        
    def get_engine(self):
        return self.engine
        



