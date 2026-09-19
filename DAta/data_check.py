import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_excel("Concrete_Data.xls")
data.columns = ["cement", "slag", "ash", "water", "superplasticizer", "coarse_aggregate", "fine_aggregate", "age", "strength"]
print("Nulls in data:", data.isna().sum())

X = data.drop(columns=["strength"]).values
y = data[["strength"]].values

X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.50, random_state=42)

print("Train/val/test sizes:", len(X_train), len(X_val), len(X_test))

feature_names = data.columns[:-1]

val_data = pd.DataFrame(X_val, columns=feature_names)
val_data["strength"] = y_val
val_data.to_excel("Validation.xlsx", index=False)

test_data = pd.DataFrame(X_test, columns=feature_names)
test_data["strength"] = y_test
test_data.to_excel("Test.xlsx", index=False)

train_data = pd.DataFrame(X_train, columns=feature_names)
train_data["strength"] = y_train
train_data.to_excel("Train.xlsx", index=False)

scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

