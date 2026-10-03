from pathlib import Path
import importlib.util
import pickle
import pkgutil

# Python 3.14 removed pkgutil.find_loader, but some older dependencies still
# call it while importing. Keep the old API available during application startup.
if not hasattr(pkgutil, "find_loader"):
    def find_loader(fullname, path=None):
        spec = importlib.util.find_spec(fullname, path)
        return None if spec is None else spec.loader

    pkgutil.find_loader = find_loader

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, callback, dcc, html
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Cars.csv"
OLD_MODEL_PATH = BASE_DIR / "car_price_model.pkl"
OLD_ENCODER_PATH = BASE_DIR / "encoder.pkl"
NEW_MODEL_PATH = BASE_DIR / "a2_car_price_model.pkl"
A3_MODEL_PATH = BASE_DIR / "model" / "a3_model.pkl"   # saved by the A3 notebook (section 3e)

# A3: the classifier was pickled from app/code/model.py, so that module must be
# importable before the pickle is loaded. It sits next to this file.
import sys
sys.path.insert(0, str(BASE_DIR))
import model  # noqa: F401,E402  (defines LogisticRegression + CarPriceClassifier)

NUMERIC_FEATURES = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
CATEGORICAL_FEATURES = ["fuel", "seller_type", "transmission", "brand"]
OPTIONAL_FEATURES = ["mileage", "engine", "max_power", "seats"]
OWNER_MAPPING = {
    "First Owner": 1,
    "Second Owner": 2,
    "Third Owner": 3,
    "Fourth & Above Owner": 4,
}
REQUIRED_LABELS = {
    "year": "Year",
    "km_driven": "Kilometers driven",
    "fuel": "Fuel",
    "seller_type": "Seller type",
    "transmission": "Transmission",
    "owner": "Owner",
    "brand": "Brand",
}


class LinearRegression(object):
    def __init__(
        self,
        regularization,
        lr=0.001,
        method="batch",
        num_epochs=500,
        batch_size=50,
        cv=None,
        init_method="zeros",
        use_momentum=False,
        momentum=0.9,
    ):
        self.lr = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method = method
        self.cv = cv
        self.regularization = regularization
        self.init_method = init_method
        self.use_momentum = use_momentum
        self.momentum = momentum

    def mse(self, ytrue, ypred):
        return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]

    def r2(self, ytrue, ypred):
        ss_res = ((ytrue - ypred) ** 2).sum()
        ss_tot = ((ytrue - ytrue.mean()) ** 2).sum()
        return 1 - (ss_res / ss_tot)

    def predict(self, X):
        return X @ self.theta

    def _coef(self):
        return self.theta[1:]

    def _bias(self):
        return self.theta[0]


class NoPenalty:
    def __call__(self, theta):
        return 0

    def derivation(self, theta):
        return np.zeros_like(theta)


class Normal(LinearRegression):
    def __init__(self, method, lr, init_method="zeros", use_momentum=False, momentum=0.9):
        self.regularization = NoPenalty()
        super().__init__(
            self.regularization,
            lr,
            method,
            init_method=init_method,
            use_momentum=use_momentum,
            momentum=momentum,
        )


class ModelBundleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in {"LinearRegression", "Normal", "NoPenalty"}:
            return globals()[name]
        return super().find_class(module, name)


def clean_training_data(drop_duplicates=False):
    df = pd.read_csv(DATA_PATH)
    if drop_duplicates:
        # A3 notebook removes exact duplicate rows first, on the raw data
        df = df.drop_duplicates().reset_index(drop=True)
    df = df[df["fuel"].isin(["Diesel", "Petrol"])].reset_index(drop=True)
    df["mileage"] = pd.to_numeric(df["mileage"].astype(str).str.split(" ").str[0], errors="coerce")
    df["engine"] = pd.to_numeric(df["engine"].astype(str).str.split(" ").str[0], errors="coerce")
    df["max_power"] = pd.to_numeric(df["max_power"].astype(str).str.split(" ").str[0], errors="coerce")
    df["brand"] = df["name"].astype(str).str.split(" ").str[0]
    df["owner"] = df["owner"].map({**OWNER_MAPPING, "Test Drive Car": 5})
    df = df[df["owner"] != 5].reset_index(drop=True)
    return df.drop(columns=["name", "torque"])


def load_pickle(path, safe_bundle=False):
    with open(path, "rb") as file:
        if safe_bundle:
            return ModelBundleUnpickler(file).load()
        return pickle.load(file)


def load_assets():
    df = clean_training_data()
    X = df.drop(columns=["selling_price"])
    y = np.log(df["selling_price"])
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    defaults = {
        "numeric": {col: float(X_train[col].median()) for col in NUMERIC_FEATURES},
        "categorical": {col: X_train[col].mode().iloc[0] for col in CATEGORICAL_FEATURES},
        "owner": int(X_train["owner"].mode().iloc[0]),
    }

    choices = {
        "fuel": sorted(X_train["fuel"].dropna().unique()),
        "seller_type": sorted(X_train["seller_type"].dropna().unique()),
        "transmission": sorted(X_train["transmission"].dropna().unique()),
        "brand": sorted(X_train["brand"].dropna().unique()),
    }

    old_model = load_pickle(OLD_MODEL_PATH)
    old_encoder = load_pickle(OLD_ENCODER_PATH)
    new_bundle = load_pickle(NEW_MODEL_PATH, safe_bundle=True)
    a3_model = load_pickle(A3_MODEL_PATH)

    return old_model, old_encoder, new_bundle, a3_model, defaults, choices


def load_a3_defaults():
    """Fill values for blank optional fields on the A3 page.

    Recreates the A3 notebook's split (duplicates dropped, 4 price classes,
    stratified 80/20 split, random_state=42) so the medians match what the
    classifier was trained with.
    """
    df = clean_training_data(drop_duplicates=True)
    df["price_class"] = pd.qcut(df["selling_price"], q=4, labels=[0, 1, 2, 3]).astype(int)
    X = df.drop(columns=["selling_price", "price_class"])
    X_train, _, _, _ = train_test_split(X, df["price_class"], test_size=0.2,
                                        random_state=42, stratify=df["price_class"])
    return {
        "numeric": {col: float(X_train[col].median()) for col in NUMERIC_FEATURES},
        "categorical": {col: X_train[col].mode().iloc[0] for col in CATEGORICAL_FEATURES},
        "owner": int(X_train["owner"].mode().iloc[0]),
    }


old_model, old_encoder, new_bundle, a3_model, defaults, choices = load_assets()
a3_defaults = load_a3_defaults()

# Price range covered by each A3 class (the qcut edges saved with the model)
A3_CLASS_RANGES = {
    c: (float(a3_model.price_bins[c]), float(a3_model.price_bins[c + 1])) for c in range(4)
}
A3_CLASS_NAMES = {0: "Budget", 1: "Lower-mid", 2: "Upper-mid", 3: "Premium"}
old_encoded_columns = old_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
new_encoded_columns = new_bundle["encoder"].get_feature_names_out(CATEGORICAL_FEATURES)

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


def nav_link(label, href, active_path):
    class_name = "nav-link active" if href == active_path else "nav-link"
    return dcc.Link(label, href=href, className=class_name)


def nav(active_path="/"):
    return html.Nav(
        className="top-nav",
        children=[
            dcc.Link(
                [html.Span("Car price", className="brand-mark"), html.Span("predictor", className="brand-tail")],
                href="/",
                className="brand-link",
            ),
            html.Div(
                className="nav-menu",
                children=[
                    nav_link("Home", "/", active_path),
                    nav_link("Old model", "/old-model", active_path),
                    nav_link("New model", "/new-model", active_path),
                    nav_link("Price class", "/a3-model", active_path),
                ],
            ),
        ],
    )


def page_shell(active_path, content):
    return html.Div(className="page-shell", children=[nav(active_path), html.Main(content)])


def field_label(label, optional=False):
    tag = html.Span("optional", className="tag-optional") if optional else None
    children = [label]
    if tag is not None:
        children.append(tag)
    return html.Span(children, className="field-label")


def number_input(prefix, field_id, label, placeholder, step="any", min_value=None, optional=False):
    return html.Label(
        className="field",
        children=[
            field_label(label, optional),
            dcc.Input(
                id=f"{prefix}-{field_id}",
                type="number",
                placeholder=placeholder,
                step=step,
                min=min_value,
                required=not optional,
                className="control",
            ),
        ],
    )


def dropdown(prefix, field_id, label, options, placeholder, optional=False):
    return html.Label(
        className="field",
        children=[
            field_label(label, optional),
            dcc.Dropdown(
                id=f"{prefix}-{field_id}",
                options=[{"label": option, "value": option} for option in options],
                placeholder=placeholder,
                clearable=optional,
                className="dropdown",
            ),
        ],
    )


def predictor_form(prefix):
    return html.Div(
        className="form-sections",
        children=[
            html.Div(
                [
                    html.Div([html.Span("01", className="section-number"), html.H3("Listing basics")], className="section-title"),
                    html.Div(
                        [
                            dropdown(prefix, "brand", "Make", choices["brand"], "Select make"),
                            number_input(prefix, "year", "Year", "2018", step=1, min_value=1980),
                            number_input(prefix, "km_driven", "Kilometres driven", "45000", step=1000, min_value=0),
                            dropdown(prefix, "owner", "Ownership", list(OWNER_MAPPING.keys()), "Select owner"),
                        ],
                        className="form-grid form-grid-four",
                    ),
                ],
                className="form-section",
            ),
            html.Div(
                [
                    html.Div([html.Span("02", className="section-number"), html.H3("Sale details")], className="section-title"),
                    html.Div(
                        [
                            dropdown(prefix, "fuel", "Fuel", choices["fuel"], "Select fuel"),
                            dropdown(prefix, "transmission", "Transmission", choices["transmission"], "Select transmission"),
                            dropdown(prefix, "seller_type", "Seller type", choices["seller_type"], "Select seller"),
                        ],
                        className="form-grid form-grid-three",
                    ),
                ],
                className="form-section",
            ),
            html.Div(
                [
                    html.Div([html.Span("03", className="section-number"), html.H3("Specifications"), html.Span("Optional", className="section-optional")], className="section-title"),
                    html.Div(
                        [
                            number_input(prefix, "mileage", "Mileage (kmpl)", "18.5", min_value=0, optional=True),
                            number_input(prefix, "engine", "Engine (CC)", "1197", step=1, min_value=0, optional=True),
                            number_input(prefix, "max_power", "Power (bhp)", "82", min_value=0, optional=True),
                            number_input(prefix, "seats", "Seats", "5", step=1, min_value=1, optional=True),
                        ],
                        className="form-grid form-grid-four",
                    ),
                ],
                className="form-section",
            ),
        ],
    )


def home_page():
    return page_shell(
        "/",
        html.Div(
            className="home-layout",
            children=[
                html.Section(
                    className="home-header",
                    children=[
                        html.Div("USED CAR / PRICE CHECK", className="home-kicker"),
                        html.H1("Car price predictor"),
                        html.P(
                            "Enter a few details from a used-car listing and compare three models built from the same dataset.",
                            className="home-lede",
                        ),
                    ],
                ),
                html.Div(
                    className="home-workbench",
                    children=[
                        html.Aside(
                            className="home-aside",
                            children=[
                                html.Div("HOW IT WORKS", className="aside-kicker"),
                                html.Ol(
                                    [
                                        html.Li([
                                            html.Strong("Use the listing"),
                                            html.Span("Enter the year, kilometres, make, fuel and sale details shown in the advert."),
                                        ]),
                                        html.Li([
                                            html.Strong("Leave gaps when needed"),
                                            html.Span("Mileage, engine, power and seats are optional. The model fills missing values from its training data."),
                                        ]),
                                        html.Li([
                                            html.Strong("Read the estimate"),
                                            html.Span("The result is an estimated selling price, not a guaranteed offer."),
                                        ]),
                                    ],
                                    className="process-list",
                                ),
                                html.Div(
                                    [
                                        html.Strong("What is being compared?"),
                                        html.P("Original model: first pipeline. Corrected model: cleaner encoding and scaling."),
                                        html.P("Price class model: predicts which of four price ranges the car falls into."),
                                    ],
                                    className="aside-note",
                                ),
                            ],
                        ),
                        html.Section(
                            className="model-choices",
                            children=[
                                html.Div(
                                    [html.H2("Choose an estimate"), html.Span("3 available", className="section-count")],
                                    className="workbench-heading",
                                ),
                                dcc.Link(
                                    html.Div(
                                        className="choice-row choice-old",
                                        children=[
                                            html.Span("01", className="choice-index"),
                                            html.Div(
                                                [html.H3("Original model"), html.P("Random forest · first cleaning pass")],
                                                className="choice-content",
                                            ),
                                            html.Span("Open  →", className="choice-action"),
                                        ],
                                    ),
                                    href="/old-model",
                                    className="choice-link",
                                ),
                                dcc.Link(
                                    html.Div(
                                        className="choice-row choice-new",
                                        children=[
                                            html.Span("02", className="choice-index"),
                                            html.Div(
                                                [html.H3("Corrected model"), html.P("Linear regression · A2 final pipeline")],
                                                className="choice-content",
                                            ),
                                            html.Span("Open  →", className="choice-action"),
                                        ],
                                    ),
                                    href="/new-model",
                                    className="choice-link",
                                ),
                                dcc.Link(
                                    html.Div(
                                        className="choice-row choice-a3",
                                        children=[
                                            html.Span("03", className="choice-index"),
                                            html.Div(
                                                [html.H3("Price class model"), html.P("Logistic regression · A3 classifier, 4 price ranges")],
                                                className="choice-content",
                                            ),
                                            html.Span("Open  →", className="choice-action"),
                                        ],
                                    ),
                                    href="/a3-model",
                                    className="choice-link",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def predictor_page(prefix, title, copy, result_note, active_path,
                   result_label="Estimated selling price", button_text="Estimate price", extra=None):
    return page_shell(
        active_path,
        html.Div(
            className=f"app-frame theme-{prefix}",
            children=[
                html.Div(
                    [
                        dcc.Link("← Back to models", href="/", className="back-link"),
                        html.Span("LIVE ESTIMATE", className="page-status"),
                    ],
                    className="page-toolbar",
                ),
                html.Div(
                    className="intro",
                    children=[
                        html.H1(title),
                        html.P(copy, className="intro-copy"),
                        *([extra] if extra is not None else []),
                    ],
                ),
                html.Div(
                    className="workspace",
                    children=[
                        html.Div(
                            className="form-panel",
                            children=[
                                html.Div(
                                    [
                                        html.H2("Vehicle details"),
                                        html.P("Use the values from the listing or registration papers."),
                                    ],
                                    className="panel-heading",
                                ),
                                predictor_form(prefix),
                                html.Button(button_text, id=f"{prefix}-predict-button", n_clicks=0, className="submit-button"),
                            ],
                        ),
                        html.Div(
                            className="result-panel",
                            children=[
                                html.Div("MODEL OUTPUT", className="result-kicker"),
                                html.Div(result_label, className="result-label"),
                                html.Div(id=f"{prefix}-prediction-output", className="prediction-output"),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    )


def old_model_page():
    return predictor_page(
        "old",
        "The first estimate",
        "This is the original model, kept running here so you can compare it against the newer one.",
        "Random forest, trained on the first version of the cleaned data.",
        "/old-model",
    )


def new_model_page():
    return predictor_page(
        "new",
        "The corrected estimate",
        "A linear regression model built from scratch, with training-set medians for missing "
        "values, proper handling of the encoded categories, and a log-scaled price target.",
        "Linear model, trained with stochastic gradient descent and momentum on the corrected data.",
        "/new-model",
    )


def a3_class_table():
    """The four price ranges, shown under the A3 page intro."""
    rows = [
        html.Li([
            html.Strong(f"Class {c} · {A3_CLASS_NAMES[c]}"),
            html.Span(f"{low:,.0f} – {high:,.0f}"),
        ])
        for c, (low, high) in A3_CLASS_RANGES.items()
    ]
    return html.Ul(rows, className="class-ranges")


def a3_model_page():
    return predictor_page(
        "a3",
        "The price class estimate",
        "Instead of one exact price, this model tells you which of four price ranges the car belongs to. "
        "Each range holds about a quarter of the cars in the dataset. A range is easier to trust than a "
        "single figure, and it is what you usually need when deciding what to list a car for or what to offer. "
        "The model is a multinomial logistic regression built from scratch, chosen after comparing training "
        "methods, learning rates and a Ridge penalty in MLflow.",
        "Logistic regression, A3 final model.",
        "/a3-model",
        result_label="Predicted price class",
        button_text="Predict price class",
        extra=a3_class_table(),
    )


app.layout = html.Div([dcc.Location(id="url"), html.Div(id="page-content")])


@callback(Output("page-content", "children"), Input("url", "pathname"))
def route_page(pathname):
    if pathname == "/old-model":
        return old_model_page()
    if pathname == "/new-model":
        return new_model_page()
    if pathname == "/a3-model":
        return a3_model_page()
    return home_page()


def validate_required(year, km_driven, fuel, seller_type, transmission, owner, brand):
    required_values = {
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
        "brand": brand,
    }
    return [REQUIRED_LABELS[field] for field, value in required_values.items() if value in (None, "")]


def make_raw_input(year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, brand,
                   fill=None):
    # fill = which training-set medians/modes to use for blanks (A1/A2 split by default, A3 split for the A3 page)
    defaults = fill if fill is not None else globals()["defaults"]
    numeric_values = {
        "year": year,
        "km_driven": km_driven,
        "mileage": mileage,
        "engine": engine,
        "max_power": max_power,
        "seats": seats,
    }
    numeric_values = {
        col: defaults["numeric"][col] if value in (None, "") else float(value)
        for col, value in numeric_values.items()
    }
    category_values = {
        "fuel": fuel or defaults["categorical"]["fuel"],
        "seller_type": seller_type or defaults["categorical"]["seller_type"],
        "transmission": transmission or defaults["categorical"]["transmission"],
        "brand": brand or defaults["categorical"]["brand"],
    }
    owner_value = OWNER_MAPPING.get(owner, defaults["owner"])
    return pd.DataFrame([{**numeric_values, "owner": owner_value, **category_values}])


def validation_message(missing):
    return html.Div(
        [
            html.Div("A few fields are still empty.", className="validation-title"),
            html.Div("Fill in: " + ", ".join(missing), className="validation-message"),
        ]
    )


def price_result(price, note):
    return html.Div(
        [
            html.Div(f"{price:,.0f}", className="price"),
            html.Div(note, className="result-note"),
        ]
    )


def predict_old(raw_input):
    encoded = old_encoder.transform(raw_input[CATEGORICAL_FEATURES])
    encoded_df = pd.DataFrame(encoded, columns=old_encoded_columns, index=raw_input.index)
    model_input = pd.concat([raw_input.drop(columns=CATEGORICAL_FEATURES), encoded_df], axis=1)
    if hasattr(old_model, "feature_names_in_"):
        model_input = model_input.reindex(columns=old_model.feature_names_in_, fill_value=0)
    return float(np.exp(old_model.predict(model_input)[0]))


def predict_new(raw_input):
    numeric_cols = new_bundle["numeric_cols"]
    model_input = raw_input.copy()
    model_input[numeric_cols] = new_bundle["scaler"].transform(model_input[numeric_cols])

    encoded = new_bundle["encoder"].transform(model_input[CATEGORICAL_FEATURES])
    encoded_df = pd.DataFrame(encoded, columns=new_encoded_columns, index=model_input.index)
    model_input = pd.concat([model_input.drop(columns=CATEGORICAL_FEATURES), encoded_df], axis=1)
    model_input = model_input.drop(columns=new_bundle["dropped_dummy_cols"], errors="ignore")
    model_input = model_input.reindex(columns=new_bundle["feature_names"], fill_value=0)

    model_array = np.concatenate((np.ones((model_input.shape[0], 1)), model_input.to_numpy()), axis=1)
    log_price = new_bundle["model"].predict(model_array)[0]
    return float(np.exp(log_price))


def predict_a3(raw_input):
    # CarPriceClassifier does its own encoding, scaling, column ordering and intercept,
    # then returns the class (0-3) from the logistic regression.
    return int(a3_model.predict(None, raw_input)[0])


def class_result(price_class, note):
    low, high = A3_CLASS_RANGES[price_class]
    return html.Div(
        [
            html.Div(f"Class {price_class} · {A3_CLASS_NAMES[price_class]}", className="price price-class"),
            html.Div(f"Roughly {low:,.0f} – {high:,.0f}", className="class-range"),
            html.Div(note, className="result-note"),
        ]
    )


def register_prediction_callback(prefix, output_id, button_id, predict_fn, note,
                                 render=price_result, fill=None, empty_text="Fill in the form and estimate a price."):
    @callback(
        Output(output_id, "children"),
        Input(button_id, "n_clicks"),
        State(f"{prefix}-year", "value"),
        State(f"{prefix}-km_driven", "value"),
        State(f"{prefix}-fuel", "value"),
        State(f"{prefix}-seller_type", "value"),
        State(f"{prefix}-transmission", "value"),
        State(f"{prefix}-owner", "value"),
        State(f"{prefix}-mileage", "value"),
        State(f"{prefix}-engine", "value"),
        State(f"{prefix}-max_power", "value"),
        State(f"{prefix}-seats", "value"),
        State(f"{prefix}-brand", "value"),
    )
    def predict_price(n_clicks, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, brand):
        if not n_clicks:
            return html.Span(empty_text, className="muted-result")

        missing = validate_required(year, km_driven, fuel, seller_type, transmission, owner, brand)
        if missing:
            return validation_message(missing)

        raw_input = make_raw_input(year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats, brand,
                                   fill=fill)
        try:
            return render(predict_fn(raw_input), note)
        except Exception as exc:
            return html.Div(
                [
                    html.Div("Couldn't make a prediction.", className="validation-title"),
                    html.Div(str(exc), className="validation-message"),
                ]
            )


register_prediction_callback(
    "old",
    "old-prediction-output",
    "old-predict-button",
    predict_old,
    "Based on the original random forest model.",
)
register_prediction_callback(
    "new",
    "new-prediction-output",
    "new-predict-button",
    predict_new,
    "Based on the saved A2 model bundle.",
)
register_prediction_callback(
    "a3",
    "a3-prediction-output",
    "a3-predict-button",
    predict_a3,
    "Based on the A3 logistic regression classifier.",
    render=class_result,
    fill=a3_defaults,
    empty_text="Fill in the form and predict a price class.",
)


app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Car Price Predictor</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Fugaz+One&family=Work+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
            :root {
                --ink: #1C231F;
                --muted: #6B6152;
                --line: #D9CBB0;
                --paper: #EAE2CF;
                --panel: #FBF8F0;
                --field: #F3EEE0;
                --brass: #9C6B2E;
                --brass-dark: #7A5222;
                --teal: #1F6E5C;
                --teal-dark: #164E42;
                --rust: #A34632;
            }

            * { box-sizing: border-box; }

            body {
                margin: 0;
                color: var(--ink);
                background: #ffffff;
                font-family: "Work Sans", Arial, sans-serif;
            }

            a { color: inherit; text-decoration: none; }

            h1, h2, h3 {
                font-family: "Fugaz One", Arial, sans-serif;
                font-weight: 400;
                letter-spacing: 0;
            }

            .page-shell { background: #ffffff; }

            .brand-link,
            .choice-card,
            .control,
            .Select-control,
            .submit-button { font-family: "Work Sans", Arial, sans-serif; }

            .page-shell {
                min-height: 100vh;
                padding: 18px 20px 40px;
            }

            .top-nav {
                width: min(1080px, 100%);
                margin: 0 auto 28px;
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 18px;
                padding: 16px 4px;
                border-bottom: 2px solid var(--ink);
            }

            .brand-link {
                display: flex;
                align-items: baseline;
                font-family: "Fugaz One", Arial, sans-serif;
                font-size: 21px;
                font-weight: 400;
            }

            .brand-mark { color: var(--ink); }
            .brand-tail { color: var(--muted); }

            .nav-menu {
                display: flex;
                align-items: center;
                gap: 4px;
                flex-wrap: wrap;
            }

            .nav-link {
                min-height: 36px;
                display: inline-flex;
                align-items: center;
                padding: 0 14px;
                border-radius: 3px;
                color: var(--muted);
                font-size: 14px;
                font-weight: 500;
            }

            .nav-link:hover { color: var(--ink); }

            .nav-link.active {
                background: var(--ink);
                color: var(--paper);
            }

            .app-frame,
            .home-grid {
                width: min(1080px, 100%);
                margin: 0 auto;
            }

            .home-layout {
                width: min(1080px, 100%);
                margin: 0 auto;
            }

            .home-header {
                max-width: none;
                padding: 18px 0 28px;
                border-bottom: 1px solid var(--line);
            }

            .home-kicker,
            .aside-kicker {
                color: var(--teal-dark);
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.12em;
            }

            .home-header h1 {
                max-width: 650px;
                margin: 16px 0 12px;
                font-size: clamp(34px, 5vw, 56px);
                line-height: 1.05;
            }

            .home-lede {
                max-width: none;
                margin: 0;
                color: var(--muted);
                font-size: 17px;
                line-height: 1.55;
                white-space: nowrap;
            }

            .home-workbench {
                display: grid;
                grid-template-columns: minmax(230px, 0.58fr) minmax(0, 1.42fr);
                gap: 48px;
                padding-top: 28px;
            }

            .home-workbench {
                display: block;
            }

            .home-aside {
                max-width: 720px;
                padding: 0 0 24px;
                border-right: 0;
                border-bottom: 1px solid var(--line);
            }

            .home-workbench .model-choices {
                max-width: 720px;
                margin-top: 30px;
            }

            .workbench-heading {
                display: flex;
                align-items: baseline;
                justify-content: flex-start;
                gap: 16px;
                padding-bottom: 13px;
                border-bottom: 2px solid var(--ink);
            }

            .workbench-heading h2 {
                margin: 0;
                font-size: 19px;
            }

            .section-count {
                color: var(--muted);
                font-size: 12px;
            }

            .choice-row {
                display: grid;
                grid-template-columns: 44px minmax(0, 1fr);
                gap: 18px;
                align-items: center;
                min-height: 84px;
                padding: 14px 0;
                border-bottom: 1px solid var(--line);
                transition: padding 140ms ease, background 140ms ease;
            }

            .choice-link:hover .choice-row {
                padding-left: 12px;
                padding-right: 12px;
                background: #fafafa;
            }

            .choice-index {
                color: var(--brass-dark);
                font-size: 12px;
                font-weight: 700;
            }

            .choice-row.choice-new .choice-index { color: var(--teal-dark); }

            .choice-row h3 {
                margin: 0 0 5px;
                font-size: 17px;
                font-weight: 600;
            }

            .choice-row p {
                margin: 0;
                color: var(--muted);
                font-size: 14px;
            }

            .choice-row .choice-action {
                grid-column: 2;
                justify-self: start;
                margin: 4px 0 0;
                white-space: nowrap;
            }

            .home-aside {
                align-self: start;
                padding-right: 26px;
                border-right: 1px solid var(--line);
            }

            .process-list {
                display: grid;
                gap: 16px;
                margin: 20px 0 24px;
                padding: 0;
                list-style: none;
                counter-reset: process;
            }

            .process-list li {
                display: grid;
                gap: 5px;
                padding-left: 28px;
                counter-increment: process;
                position: relative;
            }

            .process-list li::before {
                content: "0" counter(process);
                position: absolute;
                left: 0;
                top: 1px;
                color: var(--brass-dark);
                font-size: 11px;
                font-weight: 700;
            }

            .process-list strong { font-size: 14px; }
            .process-list span { color: var(--muted); font-size: 13px; line-height: 1.4; }

            .aside-note,
            .home-footer-note {
                color: var(--muted);
                font-size: 12px;
                line-height: 1.5;
            }

            .aside-note { margin: 0; }

            .aside-note strong {
                display: block;
                margin-bottom: 6px;
                color: var(--ink);
                font-size: 13px;
            }

            .aside-note p { margin: 0; }

            .aside-note p { white-space: nowrap; }

            .home-footer-note {
                padding-top: 24px;
                margin-top: 48px;
                border-top: 1px solid var(--line);
            }

            .home-grid {
                display: grid;
                grid-template-columns: minmax(0, 0.72fr) minmax(420px, 1.28fr);
                gap: 64px;
                align-items: start;
            }

            .home-intro h1 {
                margin: 0 0 18px;
                font-size: clamp(34px, 4.4vw, 52px);
                line-height: 1.08;
                max-width: 10ch;
            }

            .intro-copy {
                max-width: 58ch;
                margin: 0;
                color: #443E33;
                font-size: 17px;
                line-height: 1.6;
            }

            .intro-note {
                margin-top: 28px;
                padding-top: 16px;
                border-top: 1px solid var(--line);
            }

            .intro-note-text {
                color: var(--muted);
                font-size: 14px;
            }

            .model-choices {
                display: grid;
                gap: 0;
                border-top: 2px solid var(--ink);
            }

            .choice-link { display: block; }

            .choice-card {
                display: grid;
                grid-template-columns: 132px minmax(0, 1fr) auto;
                grid-template-areas: "tag content action";
                gap: 18px;
                align-items: center;
                padding: 24px 0;
                border-bottom: 1px solid var(--line);
                transition: padding 140ms ease, background 140ms ease;
            }

            .choice-link:hover .choice-card {
                padding-left: 10px;
                background: rgba(251, 248, 240, 0.6);
            }

            .choice-card.choice-old { border-top-color: var(--brass); }
            .choice-card.choice-new { border-top-color: var(--teal); }

            .choice-tag {
                display: inline-block;
                grid-area: tag;
                font-size: 13px;
                font-weight: 600;
            }

            .choice-old .choice-tag { color: var(--brass-dark); }
            .choice-new .choice-tag { color: var(--teal-dark); }

            .choice-card h2 {
                margin: 0;
                font-size: 23px;
            }

            .choice-card p {
                margin: 6px 0 0;
                color: var(--muted);
                line-height: 1.5;
                font-size: 15px;
            }

            .choice-content { grid-area: content; }

            .choice-action {
                display: inline-block;
                grid-area: action;
                margin-top: 0;
                font-size: 14px;
                font-weight: 600;
                color: var(--ink);
                border-bottom: 1px solid var(--ink);
            }

            .intro { margin-bottom: 32px; }

            .page-toolbar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 28px;
                padding-bottom: 12px;
                border-bottom: 1px solid var(--line);
            }

            .back-link {
                color: var(--muted);
                font-size: 13px;
            }

            .back-link:hover { color: var(--ink); }

            .page-status {
                color: var(--teal-dark);
                font-family: Arial, Helvetica, sans-serif;
                font-size: 10px;
                letter-spacing: 0.1em;
            }

            .intro h1 {
                margin: 0 0 10px;
                font-size: clamp(28px, 3.6vw, 38px);
            }

            .intro .intro-copy { max-width: 66ch; }

            .theme-old .intro h1 { color: var(--brass-dark); }
            .theme-new .intro h1 { color: var(--teal-dark); }

            .workspace {
                display: grid;
                grid-template-columns: minmax(0, 1fr) 340px;
                gap: 22px;
                align-items: start;
            }

            .form-panel {
                background: #ffffff;
                border: 1px solid var(--line);
                padding: 26px;
            }

            .result-panel {
                background: var(--ink);
                color: var(--paper);
                padding: 26px;
                position: sticky;
                top: 24px;
                box-shadow: 6px 6px 0 var(--line);
                overflow: hidden;
            }

            .theme-old .result-panel { box-shadow: 6px 6px 0 var(--brass); }
            .theme-new .result-panel { box-shadow: 6px 6px 0 var(--teal); }

            .panel-heading {
                margin-bottom: 20px;
                padding-bottom: 14px;
                border-bottom: 1px solid var(--line);
            }

            .panel-heading h2 {
                margin: 0 0 4px;
                font-size: 19px;
            }

            .panel-heading p {
                margin: 0;
                color: var(--muted);
                font-size: 13px;
                white-space: nowrap;
            }

            .form-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 16px;
            }

            .form-sections { display: grid; gap: 26px; }

            .form-section + .form-section {
                padding-top: 24px;
                border-top: 1px solid var(--line);
            }

            .section-title {
                display: flex;
                align-items: baseline;
                gap: 9px;
                margin-bottom: 14px;
            }

            .section-title h3 {
                margin: 0;
                font-size: 14px;
                font-weight: 600;
            }

            .section-number {
                color: var(--brass-dark);
                font-family: Arial, Helvetica, sans-serif;
                font-size: 11px;
            }

            .section-optional {
                margin-left: auto;
                color: var(--muted);
                font-size: 11px;
            }

            .form-grid-four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
            .form-grid-three { grid-template-columns: repeat(3, minmax(0, 1fr)); }

            .field {
                display: grid;
                gap: 6px;
                min-width: 0;
            }

            .field-label {
                display: flex;
                align-items: baseline;
                gap: 6px;
                color: #35414A;
                font-size: 13px;
                font-weight: 500;
            }

            .tag-optional {
                color: var(--muted);
                font-size: 12px;
                font-weight: 400;
            }

            .control,
            .Select-control {
                width: 100%;
                min-height: 42px;
                border: 1px solid var(--line) !important;
                border-radius: 3px !important;
                background: var(--field) !important;
                box-shadow: none !important;
                font-size: 15px;
                font-family: Arial, Helvetica, sans-serif;
            }

            .control {
                padding: 0 12px;
                color: var(--ink);
            }

            .control:focus,
            .Select.is-focused > .Select-control {
                border-color: var(--ink) !important;
                outline: 2px solid var(--ink);
                outline-offset: 1px;
            }

            .Select-placeholder,
            .Select-value-label {
                line-height: 40px !important;
            }

            .submit-button {
                width: 100%;
                height: 46px;
                margin-top: 22px;
                border: 2px solid var(--ink);
                border-radius: 3px;
                color: var(--paper);
                background: var(--ink);
                font-size: 15px;
                font-weight: 600;
                font-family: Arial, Helvetica, sans-serif;
                cursor: pointer;
            }

            .submit-button:hover {
                background: transparent;
                color: var(--ink);
            }

            .result-label {
                color: #C9C0A8;
                font-size: 13px;
                font-weight: 500;
            }

            .result-kicker {
                color: #A9C5B8;
                font-family: Arial, Helvetica, sans-serif;
                font-size: 10px;
                letter-spacing: 0.1em;
            }

            .result-subhead {
                display: block;
                margin-bottom: 8px;
                color: var(--paper);
                font-size: 12px;
                font-weight: 600;
            }

            .result-details {
                display: grid;
                gap: 5px;
                margin-top: 26px;
                padding-top: 16px;
                border-top: 1px solid #3A4038;
            }

            .result-detail {
                color: #C9C0A8;
                font-size: 12px;
                line-height: 1.4;
            }

            .prediction-output {
                min-height: 110px;
                display: flex;
                align-items: center;
                margin-top: 14px;
            }

            .price {
                font-family: Arial, Helvetica, sans-serif;
                font-size: 46px;
                font-weight: 600;
                font-variant-numeric: tabular-nums;
                letter-spacing: 0;
            }

            .result-note {
                margin-top: 10px;
                color: #C9C0A8;
                font-size: 13px;
                line-height: 1.5;
            }

            .model-note {
                padding-top: 16px;
                margin-top: 16px;
                border-top: 1px solid #3A4038;
                color: #C9C0A8;
                font-size: 13px;
                line-height: 1.5;
            }

            .muted-result {
                color: #C9C0A8;
                font-size: 14px;
            }

            .validation-title {
                color: #F0B99A;
                font-size: 16px;
                font-weight: 600;
            }

            .validation-message {
                margin-top: 6px;
                color: #C9C0A8;
                line-height: 1.5;
                font-size: 14px;
            }

            /* ---- A3 price class page ---- */
            .choice-row.choice-a3 .choice-index { color: var(--rust); }
            .theme-a3 .intro h1 { color: var(--rust); }
            .theme-a3 .result-panel { box-shadow: 6px 6px 0 var(--rust); }

            .class-ranges {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 10px;
                max-width: 760px;
                margin: 20px 0 0;
                padding: 0;
                list-style: none;
            }

            .class-ranges li {
                display: grid;
                gap: 4px;
                padding: 10px 12px;
                border: 1px solid var(--line);
                background: var(--panel);
            }

            .class-ranges strong { font-size: 13px; }
            .class-ranges span {
                color: var(--muted);
                font-size: 13px;
                font-variant-numeric: tabular-nums;
            }

            .price-class { font-size: 34px; }

            .class-range {
                margin-top: 8px;
                color: var(--paper);
                font-size: 16px;
                font-variant-numeric: tabular-nums;
            }

            @media (max-width: 900px) {
                .class-ranges { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }

            @media (max-width: 900px) {
                .home-grid,
                .workspace {
                    grid-template-columns: 1fr;
                }

                .home-workbench { grid-template-columns: 1fr; gap: 32px; }
                .home-aside { padding: 0 0 24px; border-right: 0; border-bottom: 1px solid var(--line); }

                .result-panel {
                    position: static;
                }

                .form-grid-four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }

            @media (max-width: 680px) {
                .page-shell { padding: 16px 14px 32px; }
                .top-nav { flex-wrap: wrap; row-gap: 12px; }
                .form-panel, .result-panel, .choice-card { padding: 18px; }
                .form-grid,
                .form-grid-four,
                .form-grid-three { grid-template-columns: 1fr; }
                .home-grid { gap: 32px; }
                .choice-card { grid-template-columns: 1fr auto; grid-template-areas: "tag action" "content content"; gap: 8px 16px; }
                .choice-card p { margin-top: 0; }
                .page-toolbar { margin-bottom: 20px; }
                .home-header { padding-top: 12px; }
                .home-workbench { padding-top: 28px; }
                .home-lede,
                .aside-note p,
                .panel-heading p { white-space: normal; }
                .choice-row { grid-template-columns: 32px minmax(0, 1fr); gap: 10px; }
                .choice-row .choice-action { grid-column: 2; justify-self: start; margin-top: 4px; }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)