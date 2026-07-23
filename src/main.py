#!/usr/bin/env python3
import typer

from apps import describe, engineer, eval, preprocess, train

app = typer.Typer(pretty_exceptions_enable=False)
app.add_typer(preprocess, name="preprocess")
app.add_typer(describe, name="describe")
app.add_typer(engineer, name="engineer")
app.add_typer(eval, name="eval")
# app.add_typer(chat, name="chat")
app.add_typer(train, name="train")

if __name__ == "__main__":
    app()
