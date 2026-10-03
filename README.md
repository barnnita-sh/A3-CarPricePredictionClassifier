# A3: Predicting Car Price (Classification)

**AT82.03 Machine Learning · st126979**

This project predicts which of **four price classes** a used car belongs to, using a multinomial logistic regression built from scratch. It extends A1/A2 (regression) with:

- classification metrics built from scratch (accuracy, per-class / macro / weighted precision, recall and F1), checked against scikit-learn
- an optional Ridge (L2) penalty
- experiments tracked on the CSIM MLflow server, with the best model registered and in Staging
- a Dash web app with an A3 page, deployed through GitHub Actions CI/CD

**Live site:** https://<your-subdomain>.ml.brain.cs.ait.ac.th _(fill in)_

## Repository structure

```
.
├── st126979_A3_CarPriceClassification.ipynb   # data prep, Tasks 1–3, experiments, report
├── Cars.csv
├── README.md
├── .github/workflows/ci-cd.yml                 # test -> build & push -> deploy
└── app/
    ├── Dockerfile
    ├── docker-compose.yaml
    ├── requirements.txt
    └── code/
        ├── app.py              # Dash app (A1, A2 and A3 pages)
        ├── a3_page.py          # A3 page: layout + prediction callback
        ├── model.py            # LogisticRegression + CarPriceClassifier (written by the notebook)
        ├── test_model.py       # unit tests
        └── model/a3_model.pkl  # final model, saved by the notebook
```

## Target classes

`selling_price` is split into 4 classes with `pd.qcut` (quartiles). The price is strongly right-skewed, so equal-width bins (`pd.cut`) would put most cars in class 0. Quartiles give roughly 25% of cars per class. The exact price range of each class is printed in the notebook and shown in the app.

## Model

`LogisticRegression(k, n, method, alpha, max_iter, penalty=None | "ridge", lambda_)`

- `method`: `"batch"`, `"minibatch"` (30% of rows) or `"sto"` (one row per step)
- `penalty="ridge"` adds λ·Σθⱼ² to the loss; the intercept row is not penalised
- metrics: `accuracy`, `precision`, `recall`, `f1_score`, `support`, `macro_*`, `weighted_*`, `classification_report`

## Experiments (MLflow)

- Server: `http://mlflow.ml.brain.cs.ait.ac.th/`
- Experiment: `st126979-a3`
- Grid: 3 methods × 3 learning rates (0.01, 0.001, 0.0001) × 3 penalty settings (none, ridge λ=0.1, ridge λ=1.0) = 27 runs, ranked by validation macro F1
- Registered model: **`st126979-a3-model`**, in **Staging**

Best configuration and test results: _(fill in from the notebook)_

## Run locally

```bash
pip install -r app/requirements.txt pytest
pytest app/code -v                 # unit tests
cd app && docker compose up --build
```

`app/requirements.txt` must include `mlflow`, `dash`, `numpy`, `pandas` and `scikit-learn`, pinned to the versions the model was trained with, so the pickle loads inside Docker.

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push to `main`:

1. **test**: installs the requirements and runs the two unit tests
   - the model accepts the expected input (a raw car row becomes a 1 × n feature matrix)
   - the output has the expected shape (one label in {0, 1, 2, 3} per row)
2. **build-and-push**: builds `app/` and pushes it to Docker Hub (only if the tests pass)
3. **deploy**: SSHes into ml-brain via the jump host and runs `docker compose pull && docker compose up -d` (only if the push succeeded)

Repository secrets used: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `SSH_USERNAME`, `SSH_PRIVATE_KEY`, `JUMP_HOST`.
