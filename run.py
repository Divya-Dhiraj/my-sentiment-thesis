# run.py
import typer
import uvicorn

# --- THE FIX: Make the import path specific and point to the correct script ---
from src.pipelines import ingest_real_data

app = typer.Typer()

@app.command()
def ingest_data():
    """
    Runs the data ingestion pipeline to read the real data, create the schema,
    and populate the databases.
    """
    typer.echo("Executing the business data ingestion pipeline...")
    # --- THE FIX: Call the correct script's run function ---
    ingest_real_data.run()
    typer.echo("✅ Ingestion complete.")

@app.command()
def start_api():
    """
    Starts the FastAPI application server using Uvicorn.
    """
    typer.echo("Starting FastAPI server at http://localhost:8001...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=True)

if __name__ == "__main__":
    app()