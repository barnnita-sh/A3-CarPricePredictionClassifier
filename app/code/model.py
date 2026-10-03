import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# mlflow is only needed for logging/serving. Fall back to a plain base class so the
# app and the unit tests can still import this file if mlflow is not installed.
try:
    from mlflow.pyfunc import PythonModel
except ImportError:  # pragma: no cover
    PythonModel = object


class LogisticRegression:
    """Multinomial logistic regression (softmax), trained with gradient descent.

    Based on 02 - Multinomial Logistic Regression.ipynb, extended for A3 with:
      * classification metrics written from scratch (Task 1)
      * an optional Ridge / L2 penalty (Task 2)
      * a numerically stable softmax, a fixed seed and a verbose switch

    X must already contain an intercept column of 1s at position 0.
    Y passed to fit() is one-hot encoded, shape (m, k).
    """

    def __init__(self, k, n, method, alpha=0.001, max_iter=5000,
                 penalty=None, lambda_=0.01, random_state=42, verbose=True):
        self.k = k                    # number of classes
        self.n = n                    # number of columns in X (incl. intercept)
        self.alpha = alpha            # learning rate
        self.max_iter = max_iter
        self.method = method          # "batch", "minibatch" or "sto"
        self.penalty = penalty        # None (plain) or "ridge"
        self.lambda_ = lambda_        # ridge strength, ignored when penalty is None
        self.random_state = random_state
        self.verbose = verbose        # print the loss every 500 iterations

    # ------------------------------------------------------------------ training
    def fit(self, X, Y):
        np.random.seed(self.random_state)          # same start weights every run
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        start_time = time.time()

        if self.method == "batch":
            for i in range(self.max_iter):
                loss, grad = self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                self._log(i, loss)

        elif self.method == "minibatch":
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0] - batch_size)   # keep a full batch
                batch_X = X[ix:ix + batch_size]
                batch_Y = Y[ix:ix + batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                self._log(i, loss)

        elif self.method == "sto":
            # Visit rows without repeats until every row has been used once, then reset.
            # (The class notebook checked `i` instead of `idx` here, so rows could repeat.)
            used = set()
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while idx in used:
                    idx = np.random.randint(X.shape[0])
                used.add(idx)
                if len(used) == X.shape[0]:
                    used = set()
                loss, grad = self.gradient(X[idx, :].reshape(1, -1), Y[idx].reshape(1, -1))
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                self._log(i, loss)

        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')

        if self.verbose:
            print(f"time taken: {time.time() - start_time:.2f}s")
        return self

    def _log(self, i, loss):
        if self.verbose and i % 500 == 0:
            print(f"Loss at iteration {i}", loss)

    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        # cross-entropy; the small epsilon avoids log(0)
        loss = - np.sum(Y * np.log(h + 1e-12)) / m
        error = h - Y
        grad = self.softmax_grad(X, error)

        if self.penalty == "ridge":
            # Task 2: J(W) = -sum(y log h) + lambda * sum(W_j^2), for j >= 1.
            # Row 0 of W is the intercept, which is NOT penalised (same lesson as A2's Ridge).
            W_no_bias = self.W.copy()
            W_no_bias[0, :] = 0
            loss += self.lambda_ * np.sum(W_no_bias ** 2) / m   # divided by m only to match the logged loss scale
            grad = grad + 2 * self.lambda_ * W_no_bias           # derivative of lambda * W^2
        return loss, grad

    def softmax(self, theta_t_x):
        # subtracting the row max doesn't change the result but stops exp() overflowing
        z = theta_t_x - np.max(theta_t_x, axis=1, keepdims=True)
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return X.T @ error

    def h_theta(self, X, W):
        return self.softmax(X @ W)

    def predict_proba(self, X_test):
        return self.h_theta(X_test, self.W)

    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)

    def plot(self):
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()

    # ------------------------------------------------------------ Task 1 metrics
    def accuracy(self, ytrue, ypred):
        # correct predictions / all predictions
        return (ytrue == ypred).sum() / len(ytrue)

    def _class_counts(self, ytrue, ypred, c):
        tp = ((ytrue == c) & (ypred == c)).sum()   # predicted c, really c
        fp = ((ytrue != c) & (ypred == c)).sum()   # predicted c, really something else
        fn = ((ytrue == c) & (ypred != c)).sum()   # really c, predicted something else
        return tp, fp, fn

    def precision(self, ytrue, ypred):
        # per class: TP / (TP + FP); 0 when the class was never predicted (same as sklearn)
        scores = np.zeros(self.k)
        for c in range(self.k):
            tp, fp, fn = self._class_counts(ytrue, ypred, c)
            scores[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        return scores

    def recall(self, ytrue, ypred):
        # per class: TP / (TP + FN)
        scores = np.zeros(self.k)
        for c in range(self.k):
            tp, fp, fn = self._class_counts(ytrue, ypred, c)
            scores[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
        return scores

    def f1_score(self, ytrue, ypred):
        # per class: harmonic mean of precision and recall
        p = self.precision(ytrue, ypred)
        r = self.recall(ytrue, ypred)
        f1 = np.zeros(self.k)
        for c in range(self.k):
            f1[c] = 2 * p[c] * r[c] / (p[c] + r[c]) if (p[c] + r[c]) > 0 else 0
        return f1

    def support(self, ytrue):
        # number of TRUE samples in each class
        return np.array([(ytrue == c).sum() for c in range(self.k)])

    def macro_precision(self, ytrue, ypred):
        return self.precision(ytrue, ypred).mean()

    def macro_recall(self, ytrue, ypred):
        return self.recall(ytrue, ypred).mean()

    def macro_f1(self, ytrue, ypred):
        return self.f1_score(ytrue, ypred).mean()

    # Weighted = each class's score times its share of the true labels.
    # The shares already sum to 1, so there is no extra division by k
    # (the "/4" in the assignment's example would make it disagree with sklearn).
    def weighted_precision(self, ytrue, ypred):
        sup = self.support(ytrue)
        return (self.precision(ytrue, ypred) * sup).sum() / sup.sum()

    def weighted_recall(self, ytrue, ypred):
        sup = self.support(ytrue)
        return (self.recall(ytrue, ypred) * sup).sum() / sup.sum()

    def weighted_f1(self, ytrue, ypred):
        sup = self.support(ytrue)
        return (self.f1_score(ytrue, ypred) * sup).sum() / sup.sum()

    def classification_report(self, ytrue, ypred):
        """Same layout as sklearn's classification_report, built from the functions above."""
        ytrue, ypred = np.asarray(ytrue), np.asarray(ypred)
        p, r, f = self.precision(ytrue, ypred), self.recall(ytrue, ypred), self.f1_score(ytrue, ypred)
        sup = self.support(ytrue)
        total = sup.sum()
        rows = {str(c): [p[c], r[c], f[c], sup[c]] for c in range(self.k)}
        rows["accuracy"] = [np.nan, np.nan, self.accuracy(ytrue, ypred), total]
        rows["macro avg"] = [self.macro_precision(ytrue, ypred), self.macro_recall(ytrue, ypred),
                             self.macro_f1(ytrue, ypred), total]
        rows["weighted avg"] = [self.weighted_precision(ytrue, ypred), self.weighted_recall(ytrue, ypred),
                                self.weighted_f1(ytrue, ypred), total]
        report = pd.DataFrame.from_dict(rows, orient="index",
                                        columns=["precision", "recall", "f1-score", "support"])
        report["support"] = report["support"].astype(int)
        return report


class CarPriceClassifier(PythonModel):
    """The trained model plus its preprocessing, packaged as a single object.

    Takes raw car rows (same columns as X in the notebook) and returns price classes.
    MLflow logs/serves it, and the Dash app and the unit tests unpickle it.
    """

    def __init__(self, model, encoder, scaler, numeric_cols, categorical_cols,
                 feature_names, price_bins):
        self.model = model
        self.encoder = encoder
        self.scaler = scaler
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.feature_names = feature_names    # final column order, baseline dummies already dropped
        self.price_bins = price_bins          # qcut edges: class c = price_bins[c] .. price_bins[c+1]

    def preprocess(self, df):
        df = df.copy()
        encoded = pd.DataFrame(self.encoder.transform(df[self.categorical_cols]),
                               columns=self.encoder.get_feature_names_out(self.categorical_cols),
                               index=df.index)
        X = pd.concat([df.drop(columns=self.categorical_cols), encoded], axis=1)
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        X = X[self.feature_names]                             # same columns + order as training
        return np.c_[np.ones(len(X)), X.to_numpy(dtype=float)]  # add the intercept column

    def predict(self, context, model_input, params=None):
        return self.model.predict(self.preprocess(model_input))
