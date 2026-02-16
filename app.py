#!/usr/bin/env python3
from src import train_app
import typer

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(train_app, name='train')

if __name__ == '__main__':
    app()

