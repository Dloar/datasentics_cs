import pandas as pd
import yaml
from sqlalchemy import create_engine

class GetAllData:
    def __init__(self, config_file='config/config_db.yaml'):
        # Load database configuration from YAML file
        self.config = self._load_config(config_file)
        self.user = self.config['database']['user']
        self.password = self.config['database']['password']
        self.host = self.config['database']['host']
        self.database = self.config['database']['database']
        self.port = self.config['database']['port']
        self.engine = self._get_db_connection()

    def _load_config(self, config_file):
        # Load YAML configuration file
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    def _get_db_connection(self):
        # Create a connection engine to the MySQL database
        DATABASE_TYPE = 'mysql'
        DBAPI = 'mysqlconnector'
        engine = create_engine(f"{DATABASE_TYPE}+{DBAPI}://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
        return engine

    def load_users(self):
        # Function to load Users data with explicit columns
        query = """
            SELECT 
                user_id,
                location,
                age
            FROM users
        """
        users_df = pd.read_sql(query, con=self.engine)
        return users_df

    def load_books(self):
        # Function to load Books data with explicit columns
        query = """
            SELECT 
                isbn,
                book_title,
                book_author,
                year_of_publication,
                publisher,
                image_url_s,
                image_url_m,
                image_url_l
            FROM books
        """
        books_df = pd.read_sql(query, con=self.engine)
        return books_df

    def load_ratings(self):
        # Function to load Ratings data with explicit columns
        query = """
            SELECT 
                user_id,
                isbn,
                book_rating
            FROM ratings
        """
        ratings_df = pd.read_sql(query, con=self.engine)
        return ratings_df

    def genre_books(self):
        # Function to load Books data with explicit columns
        query = """
            SELECT 
                isbn, 
                category, 
                description
            FROM books_genre;

        """
        books_df = pd.read_sql(query, con=self.engine)
        return books_df