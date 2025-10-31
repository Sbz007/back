import os
from dotenv import load_dotenv
import psycopg2
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

def get_connection():
    return psycopg2.connect(SUPABASE_DB_URL)
