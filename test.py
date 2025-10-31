import psycopg2

DB_URL = "postgresql://postgres:m4kOH0Rl85sQ8zJf@db.vtyyobpvyadddyyulylh.supabase.co:5432/postgres"

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT NOW();")
    result = cur.fetchone()
    print("✅ Conexión exitosa. Hora del servidor:", result)
    conn.close()
except Exception as e:
    print("❌ Error de conexión:xd", e)

