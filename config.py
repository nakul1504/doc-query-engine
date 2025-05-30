import os
import pathlib
from dotenv import load_dotenv

env_mode = os.getenv("APP_ENV", "dev")
env_file = f".env.{env_mode}"
load_dotenv(env_file)

SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "")


ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRATION_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRATION_DAYS", "1"))
REFRESH_TOKEN_EXPIRATION_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRATION_DAYS", "7"))
INACTIVITY_THRESHOLD_HOURS = int(os.getenv("INACTIVITY_THRESHOLD_HOURS", "1"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en")
QA_PIPELINE_TASK = os.getenv("QA_PIPELINE_TASK", "text2text-generation")
QA_PIPELINE_MODEL = os.getenv("QA_PIPELINE_MODEL", "google/flan-t5-base")
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "BAAI/bge-large-en")
INDEX_CLEANUP_JOB_INTERVAL_HOURS = int(os.getenv("INDEX_CLEANUP_JOB_INTERVAL_HOURS", "1"))


