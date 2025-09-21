# run.py
import typer
import uvicorn
from src.pipelines import ingest_real_data
from src.utils import check_vector_db_state

app = typer.Typer()

@app.command()
def check_data():
    """
    Quickly checks if the ChromaDB vector database is populated and up-to-date.
    """
    check_vector_db_state()

# --- NEW COMMAND FOR QUICK TESTING ---
@app.command()
def ingest_data_fast():
    """
    Runs a FAST ingestion pipeline on a SMALL SUBSET (5000 rows) of the data.
    Perfect for quickly testing changes to the ingestion logic.
    """
    typer.echo("Executing the FAST business data ingestion pipeline...")
    # Call the run function with a limit of 5000 rows
    ingest_real_data.run(limit=5000)
    typer.echo("✅ FAST Ingestion complete.")


@app.command()
def ingest_data():
    """
    Runs the full data ingestion pipeline.
    """
    typer.echo("Executing the full business data ingestion pipeline...")
    ingest_real_data.run()
    typer.echo("✅ Full Ingestion complete.")

@app.command()
def start_api():
    """
    Starts the FastAPI application server using Uvicorn.
    """
    typer.echo("Starting FastAPI server at http://localhost:8001...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8001, reload=False)

if __name__ == "__main__":
    app()