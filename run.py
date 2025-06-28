# run.py
import typer
import uvicorn
from src.pipelines import ingestion

# Create the Typer app
app = typer.Typer()

@app.command()
def ingest_data():
    """
    Runs the data ingestion pipeline to read data, create embeddings,
    and populate the databases.
    """
    typer.echo("Executing data ingestion pipeline...")
    ingestion.run()
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