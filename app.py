#!/usr/bin/env python3
from src import train_app
from src.apps import preprocess, eval
import typer

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(train_app, name='train')
app.add_typer(preprocess, name='preprocess')
app.add_typer(eval, name='eval')

if __name__ == '__main__':
    app()

