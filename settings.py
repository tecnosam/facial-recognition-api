import os
from dotenv import load_dotenv


load_dotenv()


RECORDS_DIRECTORY = os.getenv("RECORDS_DIRECTORY")
MODEL_PATH = os.getenv("MODEL_PATH", './model')

