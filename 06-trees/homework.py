
import argparse
import pandas as pd
import xgboost as xgb

from statistics import mean
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True, help="Path to car_fuel_efficiency_2026.csv")
args = parser.parse_args()

# Load data and fill missing values
df = pd.read_csv(args.data).fillna(0)

# Train, validation and test split
df_full_train, df_test = train_test_split(
    df, test_size=0.2, random_state=1
)
df_train, df_val = train_test_split(
    df_full_train, test_size=0.25, random_state=1
)

y_train = df_train.fuel_efficiency_mpg
y_val = df_val.fuel_efficiency_mpg

X_train = df_train.drop("fuel_efficiency_mpg", axis=1)
X_val = df_val.drop("fuel_efficiency_mpg", axis=1)

# Convert data to dictionaries
dv = DictVectorizer(sparse=True)

X_train = dv.fit_transform(X_train.to_dict(orient="records"))
X_val = dv.transform(X_val.to_dict(orient="records"))

feature_names = dv.get_feature_names_out()


# Q1: Decision Tree with max_depth=1
print("\nQ1: Decision Tree")

tree = DecisionTreeRegressor(max_depth=1)
tree.fit(X_train, y_train)

root_feature = feature_names[tree.tree_.feature[0]]
print("Root split feature:", root_feature)


# Q2: Random Forest with 10 trees
print("\nQ2: Random Forest")

rf = RandomForestRegressor(
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)
rf.fit(X_train, y_train)

predictions = rf.predict(X_val)
print("Validation RMSE:", round(rmse(y_val, predictions), 4))


# Q3: Find the best number of trees
print("\nQ3: Number of estimators")

for n in [10, 50, 100, 150]:
    model = RandomForestRegressor(
        n_estimators=n,
        random_state=1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    score = rmse(y_val, model.predict(X_val))
    print(f"n_estimators={n}: RMSE={score:.4f}")


# Q4: Find the best max_depth
print("\nQ4: Max depth comparison")

for depth in [10, 15, 20, 25]:
    scores = []

    for n in [10, 50, 100, 150]:
        model = RandomForestRegressor(
            n_estimators=n,
            max_depth=depth,
            random_state=1,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        scores.append(rmse(y_val, model.predict(X_val)))

    print(
        f"max_depth={depth}: "
        f"RMSEs={[round(s, 4) for s in scores]}, "
        f"mean={mean(scores):.4f}"
    )


# Q5: Feature importance
print("\nQ5: Feature importance")

rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=20,
    random_state=1,
    n_jobs=-1
)
rf.fit(X_train, y_train)

importance = rf.feature_importances_

for feature in [
    "vehicle_weight",
    "horsepower",
    "acceleration",
    "engine_displacement"
]:
    index = list(feature_names).index(feature)
    print(f"{feature}: {importance[index]:.6f}")


# Q6: XGBoost eta comparison
print("\nQ6: XGBoost")

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

watchlist = [(dtrain, "train"), (dval, "val")]

for eta in [0.3, 0.1]:
    params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=watchlist
    )

    predictions = model.predict(dval)
    score = rmse(y_val, predictions)

    print(f"eta={eta}: Validation RMSE={score:.6f}")
