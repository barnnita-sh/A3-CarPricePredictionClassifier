# A3: Predicting Car Price (Classification)

In A1 and A2 I predicted the selling price of a used car as a number. In this assignment I use the same Car Price dataset, but split the price into four price classes and predict the class instead. The model is a multinomial logistic regression written from scratch.

What is new compared to A2:

- classification metrics written from scratch (accuracy, and per-class, macro and weighted precision, recall and F1), checked against scikit-learn
- an optional Ridge (L2) penalty in the logistic regression class
- experiments tracked with MLflow, with the best model registered and moved to Staging
- a new "Price class" page in the Dash web app, deployed automatically with GitHub Actions

## Repository structure

```
.
├── st126979_A3_CarPriceClassification.ipynb   # data preparation, Tasks 1 to 3, experiments and report
├── Cars.csv                                    # dataset used by the notebook
├── README.md
├── screenshots/                                # MLflow, CI/CD and web app screenshots used in the notebook
├── .github/workflows/ci-cd.yml                 # CI/CD pipeline: test, build and push, deploy
└── app/
    ├── Dockerfile
    ├── docker-compose.yaml                     # for running the app locally
    ├── requirements.txt
    └── code/
        ├── app.py                              # Dash app with the A1, A2 and A3 pages
        ├── model.py                            # LogisticRegression and CarPriceClassifier (written by the notebook)
        ├── test_model.py                       # unit tests
        ├── Cars.csv                            # used by the app for default values
        ├── car_price_model.pkl, encoder.pkl    # A1 model
        ├── a2_car_price_model.pkl              # A2 model
        └── model/a3_model.pkl                  # A3 model, saved by the notebook
```

## Target classes

`selling_price` is split into four classes with `pd.qcut`, so each class holds about a quarter of the cars:

| Class | Price range |
| --- | --- |
| 0 | 30,000 to 250,000 |
| 1 | 250,000 to 410,000 |
| 2 | 410,000 to 640,000 |
| 3 | 640,000 to 10,000,000 |

I used quantiles instead of equal-width bins (`pd.cut`) because the price is strongly right-skewed. Equal-width bins would have put almost every car in class 0.

## Model

`LogisticRegression(k, n, method, alpha, max_iter, penalty=None or "ridge", lambda_)`

- `method`: `"batch"`, `"minibatch"` (30% of the rows per step) or `"sto"` (one row per step)
- `penalty="ridge"` adds λ times the sum of squared weights to the loss. The intercept row is not penalised.
- Metrics: `accuracy`, `precision`, `recall`, `f1_score`, `support`, the `macro_*` and `weighted_*` versions, and `classification_report`. All of them match scikit-learn's output.

## Experiments (MLflow)

The course MLflow server was unstable during the assignment. Following the TA announcement on 1 October 2026, the experiments were logged to a local MLflow server (`http://localhost:5000`) and screenshots are included in the notebook. Setting `USE_CSIM_SERVER = True` in the notebook switches the code to the course server.

- Experiment: `st126979-a3`
- Grid: 3 methods x 3 learning rates (0.01, 0.001, 0.0001) x 3 penalty settings (none, ridge λ = 0.1, ridge λ = 1.0), 27 runs in total, compared on validation macro F1
- Registered model: `st126979-a3-model`, in Staging

**Best configuration:** batch gradient descent, learning rate 0.0001, ridge λ = 0.1

**Test set results:** accuracy 0.724, macro F1 0.719, weighted F1 0.724. The cheapest and most expensive classes are predicted well (F1 0.85 and 0.83). The two middle classes are harder (F1 0.61 and 0.59), and almost all mistakes are between neighbouring classes.

## Running it locally

From the repository root:

```bash
pip install -r app/requirements.txt pytest
pytest app/code -v                  # runs the two unit tests
cd app && docker compose up --build # starts the web app
```

Then open `http://localhost:8050`.

The versions of numpy, pandas and scikit-learn in `app/requirements.txt` are pinned to the ones the model was trained with, and MLflow is included, so the saved model loads the same way inside Docker.

## CI/CD

`.github/workflows/ci-cd.yml` runs on every push to `main`. It has three jobs, and each one only runs if the previous one passed:

1. **test:** installs the requirements and runs the two unit tests
   - the model takes the expected input (a raw car row becomes one row of features with the intercept)
   - the output has the expected shape (one class from 0 to 3 for each input row)
2. **build-and-push:** builds the Docker image from `app/` and pushes it to Docker Hub as `barnnitash/car-price-app:a3`
3. **deploy:** connects to ml-brain over SSH through the bazooka jump host and runs `docker compose pull` and `docker compose up -d` in `~/st126979-app`

The pipeline uses these repository secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `SSH_USERNAME`, `SSH_PRIVATE_KEY` and `JUMP_HOST`.

## Deployment

The app runs on ml-brain on port 8979.

The Traefik reverse proxy was not running on ml-brain during this assignment, so the subdomain from A2 (`web-st126979.ml.brain.cs.ait.ac.th`) could not be used. The container publishes port 8979 directly instead. Since that port is not open from outside the server, the app can be viewed through an SSH tunnel:

```bash
ssh -i ~/.ssh/st126979 -J st126979@bazooka.cs.ait.ac.th -L 8979:localhost:8979 st126979@ml.brain.cs.ait.ac.th
```

and then opening `http://localhost:8979` in the browser.

The web app has three pages:

- **Old model:** the A1 random forest
- **New model:** the A2 linear regression
- **Price class:** the A3 logistic regression, which shows the predicted class and its price range