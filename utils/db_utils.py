from supabase import create_client
import pandas as pd

SUPABASE_URL = "https://vtyyobpvyadddyyulylh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ0eXlvYnB2eWFkZGR5eXVseWxoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIxMjgyMSwiZXhwIjoyMDc1Nzg4ODIxfQ.lBI_pSwY7gZ84AbmZpnW55mHFv86BaONF9Xenfj0g28"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


def create_table(table_name: str, columns: list):
    # todas las columnas text para evitar conflictos
    cols = ", ".join([f'"{c}" text' for c in columns])
    query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id serial primary key,
        {cols}
    );
    """
    supabase.rpc("execute_sql", {"query": query}).execute()


def insert_dataframe(table_name: str, df: pd.DataFrame):
    records = df.to_dict(orient="records")
    supabase.table(table_name).insert(records).execute()
