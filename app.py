# ====================================================
# 📘 app.py — FastAPI + Supabase + PyTorch CSV Handler
# ====================================================

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import pandas as pd
import numpy as np
import io, os

from supabase import create_client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss

import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
load_dotenv()
# ====================================================
# 🔧 Configuración de Supabase
# ====================================================
SUPABASE_URL = "https://vtyyobpvyadddyyulylh.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InZ0eXlvYnB2eWFkZGR5eXVseWxoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MDIxMjgyMSwiZXhwIjoyMDc1Nzg4ODIxfQ.lBI_pSwY7gZ84AbmZpnW55mHFv86BaONF9Xenfj0g28"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ====================================================
# 🚀 Inicialización FastAPI
# ====================================================
app = FastAPI(title="CSV → Supabase Multi-Table API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================
# 🧠 Variables globales
# ====================================================
DATAFRAME_CACHE = None
last_cleaned_df = None

# ====================================================
# ⚙️ Funciones auxiliares
# ====================================================
def create_table_if_not_exists(table_name: str, columns: dict):
    """Crea tabla en Supabase si no existe."""
    columns_sql = ", ".join([f'"{col}" {tipo}' for col, tipo in columns.items()])
    query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql});'

    try:
        supabase.rpc("execute_sql", {"query": query}).execute()
        print(f"✅ Tabla '{table_name}' creada o existente")
    except Exception as e:
        print(f"❌ Error creando tabla '{table_name}':", e)


def insert_dataframe(table_name: str, df: pd.DataFrame):
    """Inserta datos en una tabla de Supabase."""
    try:
        data = df.to_dict(orient="records")
        supabase.table(table_name).insert(data).execute()
        print(f"✅ {len(df)} filas insertadas en '{table_name}'")
    except Exception as e:
        print(f"❌ Error insertando datos en '{table_name}':", e)


def clean_dataframe(df: pd.DataFrame):
    """Limpieza general: elimina duplicados y rellena nulos."""
    df = df.drop_duplicates()
    df = df.fillna("Sin valor")
    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)
    return df


def actualizar_estado_alumnos(tabla_alumnos: str, tabla_notas: str):
    """Actualiza el estado aprobado/desaprobado de los alumnos."""
    try:
        notas_res = supabase.table(tabla_notas).select("alumno_id, promedio").execute()
        notas = notas_res.data

        for n in notas:
            alumno_id = n["alumno_id"]
            promedio = float(n["promedio"])

            if promedio >= 11:
                supabase.table(tabla_alumnos).update({"aprobado": 1, "desaprobado": 0}).eq("id", alumno_id).execute()
            else:
                supabase.table(tabla_alumnos).update({"aprobado": 0, "desaprobado": 1}).eq("id", alumno_id).execute()

        print("✅ Estado de alumnos actualizado correctamente")
    except Exception as e:
        print(f"❌ Error actualizando estado de alumnos: {e}")

# ====================================================
# 🌐 Endpoints principales
# ====================================================

@app.get("/")
def root():
    return {"message": "API activa. Usa /upload_csv para subir CSV."}


# ----------------------------------------------------
# 📤 Subir y procesar CSV
# ----------------------------------------------------
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    global DATAFRAME_CACHE, last_cleaned_df

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        DATAFRAME_CACHE = df.copy()
        last_cleaned_df = df.copy()

        base_name = os.path.splitext(file.filename)[0]
        tables_inserted = []

        # ---- 🧩 Tabla Alumnos ----
        if all(col in df.columns for col in ["nombre", "apellidos", "grado", "seccion", "dni"]):
            df_alumnos = df[["nombre", "apellidos", "grado", "seccion", "dni"]].copy()
            df_alumnos["aprobado"] = 0
            df_alumnos["desaprobado"] = 0
            df_alumnos = clean_dataframe(df_alumnos)

            tabla_alumnos = f"alumnos_{base_name}"
            create_table_if_not_exists(tabla_alumnos, {col: "text" for col in df_alumnos.columns})
            insert_dataframe(tabla_alumnos, df_alumnos)
            tables_inserted.append({"table_name": tabla_alumnos, "rows": len(df_alumnos)})

        # ---- 📚 Tabla Cursos ----
        if all(col in df.columns for col in ["nombre_curso", "profesor", "anio_escolar"]):
            df_cursos = df[["nombre_curso", "profesor", "anio_escolar"]].copy()
            df_cursos.rename(columns={"nombre_curso": "nombre", "profesor": "curso", "anio_escolar": "anio"}, inplace=True)
            df_cursos = clean_dataframe(df_cursos)

            tabla_cursos = f"cursos_{base_name}"
            create_table_if_not_exists(tabla_cursos, {col: "text" for col in df_cursos.columns})
            insert_dataframe(tabla_cursos, df_cursos)
            tables_inserted.append({"table_name": tabla_cursos, "rows": len(df_cursos)})

        # ---- 🧮 Tabla Notas ----
        if all(col in df.columns for col in ["alumno_id", "curso_id", "nota_1", "nota_2", "nota_3", "nota_4"]):
            df_notas = df[["alumno_id", "curso_id", "nota_1", "nota_2", "nota_3", "nota_4"]].copy()
            df_notas["promedio"] = df_notas[["nota_1", "nota_2", "nota_3", "nota_4"]].astype(float).mean(axis=1)
            df_notas = clean_dataframe(df_notas)

            tabla_notas = f"notas_{base_name}"
            create_table_if_not_exists(
                tabla_notas,
                {col: "text" if col in ["alumno_id", "curso_id"] else "float" for col in df_notas.columns}
            )
            insert_dataframe(tabla_notas, df_notas)
            actualizar_estado_alumnos(tabla_alumnos, tabla_notas)
            tables_inserted.append({"table_name": tabla_notas, "rows": len(df_notas)})

        if not tables_inserted:
            return {"error": "❌ CSV no contiene columnas válidas para ninguna tabla."}

        return {"message": "✅ CSV procesado correctamente", "tables": tables_inserted}

    except Exception as e:
        return {"error": f"❌ Error procesando CSV: {str(e)}"}


# ----------------------------------------------------
# 🧹 Limpieza de datos
# ----------------------------------------------------
@app.post("/clean_data")
async def clean_data(payload: dict):
    global DATAFRAME_CACHE, last_cleaned_df

    if DATAFRAME_CACHE is None:
        return {"error": "❌ No hay CSV cargado"}

    action = payload.get("action", "clean_all")
    df = DATAFRAME_CACHE.copy()

    try:
        if action == "clean_all":
            df = df.drop_duplicates()
            df = df.fillna(df.select_dtypes(include=["number"]).mean())
            df = df.fillna("Sin valor")

        elif action.startswith("clean_"):
            col = action.replace("clean_", "")
            if col not in df.columns:
                return {"error": f"Columna '{col}' no encontrada."}
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Desconocido")
            elif df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].mean())
        else:
            return {"error": "Acción de limpieza no reconocida."}

        DATAFRAME_CACHE = df
        last_cleaned_df = df

        return {"message": f"✅ Limpieza '{action}' completada correctamente", "rows_after": len(df)}

    except Exception as e:
        return {"error": f"❌ Error durante la limpieza: {str(e)}"}


# ----------------------------------------------------
# 📥 Obtener datos limpios
# ----------------------------------------------------
@app.get("/get_cleaned_csv")
async def get_cleaned_csv():
    if last_cleaned_df is None:
        return JSONResponse(content={"error": "No hay datos limpios"}, status_code=404)
    return last_cleaned_df.to_dict(orient="records")


# ----------------------------------------------------
# 🧠 Entrenamiento del modelo
# ----------------------------------------------------
@app.post("/train_model")
async def train_model(payload: dict = Body(...)):
    model_type = payload.get("model_type", "classification")
    epochs = int(payload.get("epochs", 10))
    batch_size = int(payload.get("batch_size", 32))

    # 1️⃣ Cargar datos desde Supabase
    notas_res = supabase.table("notas_6toA_matematica_luis").select("*").execute()
    alumnos_res = supabase.table("alumnos_6toA_matematica_luis").select("id, aprobado").execute()

    notas_df = pd.DataFrame(notas_res.data)
    alumnos_df = pd.DataFrame(alumnos_res.data)

    # 2️⃣ Preparar datos
    df = pd.merge(notas_df, alumnos_df, left_on="alumno_id", right_on="id")
    X = df[["nota_1", "nota_2", "nota_3", "nota_4"]].astype(float).values
    y = df["aprobado"].astype(int).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3️⃣ Modelo en PyTorch
    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
        nn.Softmax(dim=1)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 4️⃣ Entrenamiento
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        pred_labels = torch.argmax(outputs, dim=1)
        acc = (pred_labels == y_train_tensor).float().mean().item()
        history.append({"epoch": epoch + 1, "loss": loss.item(), "accuracy": acc})

    # 5️⃣ Evaluación
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    test_outputs = model(X_test_tensor)
    test_pred = torch.argmax(test_outputs, dim=1)

    test_acc = accuracy_score(y_test_tensor, test_pred)
    test_loss = log_loss(y_test_tensor, test_outputs.detach().numpy())

    return {
        "model_type": model_type,
        "epochs": epochs,
        "batch_size": batch_size,
        "history": history,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
