import pandas as pd
from sklearn.model_selection import train_test_split  # pozwoli na podzielenie danych na do trenowania i do testow
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler  # standaryzuje dane
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle
from sklearn.metrics import accuracy_score

df = pd.read_csv("data.csv")

x = df.drop('class', axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

pipelines = {
    # "lr": make_pipeline(StandardScaler(), LogisticRegression()),
    # "rc": make_pipeline(StandardScaler(), RidgeClassifier()),
    "rf": make_pipeline(StandardScaler(), RandomForestClassifier()),
    # "gb": make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

fit_models = {}
print("Starting training...")
for algo, pipeline in pipelines.items():
    print(f"Training with: {algo}")
    model = pipeline.fit(x_train, y_train)
    fit_models[algo] = model


for algo, model in fit_models.items():
    print(f"predict with {algo}")
    yhat = model.predict(x_test)
    print(algo, accuracy_score(y_test, yhat))


with open(r"model.pkl", "wb") as f:
    pickle.dump(fit_models['rf'], f)
