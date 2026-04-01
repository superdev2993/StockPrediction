from numerapi import NumerAPI
import pandas as pd
import json
import matplotlib.pyplot as plt
napi = NumerAPI()


# use one of the latest data versions
DATA_VERSION = "v5.2"

# Download data
napi.download_dataset(f"{DATA_VERSION}/train.parquet")
napi.download_dataset(f"{DATA_VERSION}/features.json")

# Load data
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
features = feature_metadata["feature_sets"]["small"]
# use "medium" or "all" for better performance. Requires more RAM.
# features = feature_metadata["feature_sets"]["medium"]
# features = feature_metadata["feature_sets"]["all"]
train = pd.read_parquet(f"{DATA_VERSION}/train.parquet", columns=["era"]+features+["target"])

# For better models, join train and validation data and train on all of it.
# This would cause diagnostics to be misleading though.
# napi.download_dataset(f"{DATA_VERSION}/validation.parquet")
# validation = pd.read_parquet(f"{DATA_VERSION}/validation.parquet", columns=["era"]+features+["target"])
# validation = validation[validation["data_type"] == "validation"] # drop rows which don't have targets yet
# train = pd.concat([train, validation])

# Downsample for speed
train = train[train["era"].isin(train["era"].unique()[::4])]  # skip this step for better performance

# # Plot density histogram of the target
# train["target"].plot(
#   kind="hist",
#   title="Target",
#   figsize=(5, 3),
#   xlabel="Value",
#   density=True,
#   bins=50
# )
# plt.show()



# Train model
import lightgbm as lgb
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5-1,
    colsample_bytree=0.1
)
# We've found the following "deep" parameters perform much better, but they require much more CPU and RAM
# model = lgb.LGBMRegressor(
#     n_estimators=30_000,
#     learning_rate=0.001,
#     max_depth=10,
#     num_leaves=2**10,
#     colsample_bytree=0.1,
#     min_data_in_leaf=10000,
# )
model.fit(
    train[features],
    train["target"]
)

# Define predict function
def predict(
    live_features: pd.DataFrame,
    live_benchmark_models: pd.DataFrame
 ) -> pd.DataFrame:
    live_predictions = model.predict(live_features[features])
    submission = pd.Series(live_predictions, index=live_features.index)
    return submission.to_frame("prediction")

# Pickle predict function
import cloudpickle
p = cloudpickle.dumps(predict)
with open("example_model.pkl", "wb") as f:
    f.write(p)

# Download file if running in Google Colab
try:
    from google.colab import files
    files.download('example_model.pkl')
except:
    pass


# Download validation data - this will take a few minutes
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")

# Load the validation data and filter for data_type == "validation"
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation.parquet",
    columns=["era", "data_type", "target"] + features
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]

# Downsample to every 4th era to reduce memory usage and speedup evaluation (suggested for Colab free tier)
# Comment out the line below to use all the data (slower and higher memory usage, but more accurate evaluation)
validation = validation[validation["era"].isin(validation["era"].unique()[::4])]

# Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
# so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]

# Generate predictions against the out-of-sample validation features
# This will take a few minutes 🍵
validation["prediction"] = model.predict(validation[features])
print(validation[["era", "prediction", "target"]])


# import the 2 scoring functions
from numerai_tools.scoring import numerai_corr, correlation_contribution

# Download and join in the meta_model for the validation eras
napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
validation["meta_model"] = pd.read_parquet(
    f"v4.3/meta_model.parquet"
)["numerai_meta_model"]


# Compute the per-era corr between our predictions and the target values
per_era_corr = validation.groupby("era").apply(
    lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
)

# Compute the per-era mmc between our predictions, the meta model, and the target values
per_era_mmc = validation.dropna().groupby("era").apply(
    lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
)


# Plot the per-era correlation
per_era_corr.plot(
  title="Validation CORR",
  kind="bar",
  figsize=(8, 4),
  xticks=[],
  legend=False,
  snap=False
)
per_era_mmc.plot(
  title="Validation MMC",
  kind="bar",
  figsize=(8, 4),
  xticks=[],
  legend=False,
  snap=False
)


# Plot the cumulative per-era correlation
per_era_corr.cumsum().plot(
  title="Cumulative Validation CORR",
  kind="line",
  figsize=(8, 4),
  legend=False
)
per_era_mmc.cumsum().plot(
  title="Cumulative Validation MMC",
  kind="line",
  figsize=(8, 4),
  legend=False
)

# Compute performance metrics
corr_mean = per_era_corr.mean()
corr_std = per_era_corr.std(ddof=0)
corr_sharpe = corr_mean / corr_std
corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

mmc_mean = per_era_mmc.mean()
mmc_std = per_era_mmc.std(ddof=0)
mmc_sharpe = mmc_mean / mmc_std
mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max()

pf = pd.DataFrame({
    "mean": [corr_mean, mmc_mean],
    "std": [corr_std, mmc_std],
    "sharpe": [corr_sharpe, mmc_sharpe],
    "max_drawdown": [corr_max_drawdown, mmc_max_drawdown]
}, index=["CORR", "MMC"]).T

print(pf)


# Download latest live features
napi.download_dataset(f"{DATA_VERSION}/live.parquet")

# Load live features
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=features)

# Generate live predictions
live_predictions = model.predict(live_features[features])

# Format submission
ps = pd.Series(live_predictions, index=live_features.index).to_frame("prediction")
