# Citation Forecast

Welcome to citation-forecast, a CLI-based machine learning pipeline
designed to forecast academic impact. I built this
as a hands-on environment to experiment with NLP forecasting,
while extending my practical knowledge of 
Python, software engineering, and deep learning.

The core training loop and data pre-processing is driven entirely by
CLI and integrates directly with MLflow to track experiments and compare
iterations efficiently. I integrated  a custom registry system, allowing clean, string-based access
to models, loss functions,
and optimizers directly from configuration files, with all inputs strictly
validated by Pydantic. 

# 1. Tech Stack & Key Features
## 1.1 Key Features
* **Modular Trainer Architecture:**
Decoupled training loops using a Registry (utils) pattern for easy swapping of losses, optimizers, and models.

* **Production-Ready Inference:** 
Integrated with Modal for serverless GPU inference and MLflow for experiment tracking.

* **Data Pipeline:** 
Data pre-processing CLI and custom PyTorch Datasets/Loaders forming a flexible ETL pipeline

## 1.2 Tech Stack
* **Core:** Python (3.13), PyTorch (Deep Learning)

* **MLOps:** MLflow (tracking), Modal (Deployment), Scikit-Learn (metric calculation)

* **Data:** Polars (predicate pushdown loading), Polars (multi-worker serving)

* **Validation:** Pydantic (Type safe model config schemas)

* **Tooling:** Typer (CLI), Rich (Console logging)

# 2. Road-map

## Completed

<details> 
<summary> <b>Expand List</b></summary> 

> ## v0.2
> <details>
> <summary><b>v0.2.0 Data pre-processing CLI</b></summary>
>
> - Standardise data pre-processing notebook into separate functions.
> - Orchestrate pre-processing functions in main loop.
> - CLI control via Typer app, functions induced/skipped via option flags.
>
> </details>

> ## v0.3
> <details>
> <summary><b>v0.3.0 Dataloader Flexibility</b></summary>
> 
> - Support concatenation of multiple string/token columns when serving examples from dataloader.
> 
> </details>


> ## v0.4 
> <details>
> <summary><b>v0.4.0 Eval CLI</b></summary>
> 
> - CLI app for evaluating models
> - Config driven with optional CLI overrides 
> - Temporal scrub with variable intervals and MLflow tracking
> - JSON data export for website historical performance graphs
> 
> </details>
> 
> <details>
> <summary><b>v0.4.1 Metric Tracker Generalisation</b></summary>
> 
> - Generalise & simplify metric tracker methods/initialisation
> - Generalised parameter init function & metric calc functions responsive to store name prefix
> 
> </details>

</details>

## v0.5
<details>
<summary><b>v0.5.0 Feature Consolidation (In-Progress)</b></summary>

- Move code-as-config module from root to src, allow config value overrides from CLI via option flags
- Refactor train (main) loop into /apps
- Create metric tracker base class for tracking logic, overide metric calculation in children 
- Seperate train/data/env configs into distinct files
- Add dedicated loss/optimisation config for lr schedule milestones etc.

</details>

## v1.0
<details>
<summary><b>v1.0.0 Full experiment suite (Planned)</b></summary>

- Refactor src module imports as relative to /src, standardise cli access via pyproject.toml build parameters
- MLflow run management with automatic name creation, CLI driven checkpoint loading with MLflow child run assignment 
- Automated Hyper parameter search logic, with MLflow parent/child assignment

</details>

## v1.1
<details>
<summary><b>v1.1.0 Text Embedding (Planned)</b></summary>

- Embed string columns via pre-processing CLI

</details>

# 3. Project Structure  
```text
citation-forecast/
├── config/
│   ├── config.py               # general config
│   └── env.py                  # env variables
├── production/
│   ├── service.py              # modal inference image
│   └── models/                 # config, architechture, and weights for production models
│       └── examp-model/
│           ├── model/
│           │   ├── arch.py     # architechture and config schema
│           │   └── config.py   # hyperperameter config 
│           ├── tokeniser/      # tokeniser (transformers)
│           └── weights/        # model checkpoint (.pt)
├── src/
│   ├── apps/                   # typer CLI argument parsing
│   ├── builders/               # safety checks, registry access, and instance creations to serve assets into main loops
│   └── training/
│       ├── eval/               # evaluation loops
│       ├── losses/             # loss funcs (registered)
│       ├── optimizers/         # optimisers (registered)
│       ├── tracking/           # metric tracking, calculation, and MLflow logging 
│       └── callbacks/          # early stopping
├── models/                     # PyTorch modules with Pydantic config schema's (registered)
├── data/
│   ├── datasets/               # PyTorch datasets
│   │    └── OrdinalDataset.py  # filter/load dataset into memory, formats y as ordinal classes
│   ├── samplers/               # samplers for PyTorch Datasets
│   ├──  preprocess/            # clean/tokenise and stage selections of main dataset > sub datasets
│   └── sample/                 # sample dataset for local testing
├── utils/
│   ├── logging/                # file & console (rich formatted) logging
│   └── register/               # registry class, allows string access to objects via decorators
└── app.py                      # CLI entry point (./app.py)
```

# 4. Quick start
* Clone the repository and setup your environment.
```bash
git clone https://github.com/Felix-Noble/citation-forecast.git
cd citation-forecast
# use your preferred env manager here
``` 

* Run the MLflow tracking server 
```bash 
# activate venv containing mlflow, or use uv/pipx
mlflow server 
```

* Start a training run on cpu (model/dataset selection & configuration in config.py)
```bash
./app.py train -s demo-run --no-gpu
```
