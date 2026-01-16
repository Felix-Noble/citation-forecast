#!/usr/bin/env python3
import mlflow
import typer

app = typer.Typer()

@app.command()
def main(
    experiment = typer.Argument(
        help='experiment name'
    ),
    run_id = typer.Argument(
        help='run_id'
    ),

):
    mlflow.set_tracking_uri('http://127.0.0.1:5000') 
    mlflow.set_experiment(experiment)
    mlflow.start_run(run_id = run_id)
    mlflow.end_run()

if  __name__ == '__main__':
    app()
