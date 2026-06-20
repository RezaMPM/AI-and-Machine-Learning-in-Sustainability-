# ============================================================================
# IMPORTING THE LIBRARIES WE WILL USE
# ============================================================================

# --- pandas: the main tool for working with tables of data --------------------
# Think of it as a programmable Excel. We nickname it "pd".
import pandas as pd

# --- numpy: a library for fast maths on lists of numbers ----------------------
# Nicknamed "np". We use it for random numbers and array maths.
import numpy as np

# --- matplotlib.pyplot: the basic library for drawing charts ------------------
# We nickname this specific part "plt".
import matplotlib.pyplot as plt

# --- seaborn: makes nicer-looking statistical charts (built on matplotlib) ----
import seaborn as sns

# --- scikit-learn (sklearn): THE machine-learning toolbox ---------------------
# Here we import only the specific pieces we need.
# "from X import Y" means: from library X, bring in just the tool Y.
from sklearn.model_selection import train_test_split          # splits data into train/test parts
from sklearn.preprocessing  import StandardScaler             # puts features on the same scale
from sklearn.linear_model   import LinearRegression           # a simple prediction model
from sklearn.ensemble       import RandomForestRegressor, RandomForestClassifier  # stronger models
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,    # scoring tools
                             accuracy_score, classification_report, confusion_matrix)

# --- A fixed "random seed" for reproducible results ---------------------------
# Some steps use randomness. Fixing the seed to a number (42) means everyone in
# the class gets the SAME results every time the notebook is run.
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)        # tell numpy to use this seed

# --- Cosmetic settings (purely visual) ---------------------------------------
sns.set_style("whitegrid")          # light grid behind charts
pd.set_option("display.max_columns", None)   # show all columns of wide tables

# If everything imported correctly, the messages below will print.
print("✅ All libraries imported successfully!")
print("   pandas version:", pd.__version__)   # __version__ tells us the library's version
print("   numpy  version:", np.__version__)

# ============================================================================
#== END OF Part 1 SETUP CODE =================================================

# ============================================================================
# LOADING THE DATA (with an automatic backup if the internet is blocked)
# ============================================================================

# These are the human-readable names we will give the 8 building features.
# Square brackets [ ] create a "list" — an ordered collection of items.
feature_names = ["Relative_Compactness", "Surface_Area", "Wall_Area", "Roof_Area",
                 "Overall_Height", "Orientation", "Glazing_Area", "Glazing_Area_Distribution"]

# ---- Function 1: try to download the REAL dataset ----------------------------
# "def" starts a function definition. Everything indented below belongs to it.
def load_real():
    from ucimlrepo import fetch_ucirepo      # a helper that downloads UCI datasets
    ds = fetch_ucirepo(id=242)               # 242 is the ID of the Energy Efficiency dataset
    X = ds.data.features.copy()              # .copy() makes an independent copy of the table
    X.columns = feature_names                # rename the columns to our readable names
    y = ds.data.targets.iloc[:, 0].rename("Heating_Load")   # take the first target column
    df = pd.concat([X, y], axis=1)           # glue features (X) and target (y) side by side
    return df, "UCI repository (real data)"  # a function "returns" its result to whoever called it

# ---- Function 2: build a synthetic stand-in if the download fails ------------
def load_synthetic(n=768):                   # n=768 is a default: 768 rows unless we say otherwise
    rng = np.random.default_rng(RANDOM_STATE)        # a random-number generator
    rc   = rng.uniform(0.62, 0.98, n)        # 'n' random compactness values between 0.62 and 0.98
    sa   = rng.uniform(514, 808, n)          # surface area
    wa   = rng.uniform(245, 416, n)          # wall area
    ra   = rng.uniform(110, 220, n)          # roof area
    oh   = rng.choice([3.5, 7.0], n)         # overall height: pick one of two typical values
    ori  = rng.integers(2, 6, n)             # orientation: whole numbers 2..5
    gla  = rng.choice([0.0, 0.10, 0.25, 0.40], n)    # glazing area: one of these options
    glad = rng.integers(0, 6, n)             # glazing distribution: 0..5
    # A simple formula linking features to heating load, plus a bit of random "noise":
    heating = (35*rc + 0.02*wa - 0.05*ra + 2.2*oh + 18*gla
               - 0.01*sa + rng.normal(0, 2.0, n))
    heating = np.clip(heating, 6, 43)        # keep values inside the realistic 6..43 range
    # np.column_stack puts these arrays together as columns of one table:
    df = pd.DataFrame(np.column_stack([rc, sa, wa, ra, oh, ori, gla, glad]),
                      columns=feature_names)
    df["Heating_Load"] = heating             # add the target column
    return df, "synthetic fallback (no internet)"

# ---- Try the real download first; fall back to synthetic if anything fails ---
try:
    import subprocess, sys
    # Quietly install the 'ucimlrepo' helper (the '-q' means "quiet"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ucimlrepo"], check=False)
    df, source = load_real()                 # if this line errors, Python jumps to 'except'
except Exception as e:
    # This block runs ONLY if the 'try' block failed.
    print(f"⚠️  Could not load the real data ({e.__class__.__name__}). Using synthetic data instead.")
    df, source = load_synthetic()

# f"..." is an "f-string": text with {values} inserted into it.
print(f"✅ Data ready — source: {source}")
print(f"   The table has {df.shape[0]} rows (buildings) and {df.shape[1]} columns.")
# (df.shape gives (rows, columns); [0] picks the first number, [1] the second.)

# Show the first 5 rows. Because this is the last line in the cell, Colab
# displays the table nicely underneath automatically.
print(df.head())

# Summary statistics for every column, rounded to 2 decimal places.
# Reading left to right: count of values, mean (average), std (spread), min, max, etc.
print(df.describe().round(2))

# ============================================================================
#== END OF Part 2 SETUP CODE =================================================

# ---- Chart 1: histogram of the target -------------------------------------
plt.figure(figsize=(7, 4))                 # start a new chart, 7x4 inches in size
sns.histplot(df["Heating_Load"], kde=True, color="#2F5496")
#   df["Heating_Load"]  -> selects just that one column (square brackets pick a column by name)
#   kde=True            -> also draw a smooth curve over the bars
#   color="#2F5496"     -> a navy colour (hex code)
plt.title("Distribution of Heating Load")  # chart title
plt.xlabel("Heating Load (kWh/m²)")        # label for the horizontal axis
plt.ylabel("Number of buildings")          # label for the vertical axis
plt.show()                                 # display the finished chart


# ---- Chart 2: correlation heatmap -----------------------------------------
# Correlation is a number from -1 to +1 describing how two columns move together:
#   +1 = perfectly together, 0 = unrelated, -1 = perfectly opposite.
plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(numeric_only=True),    # df.corr() computes correlations between all columns
            annot=True,                    # write the number inside each square
            fmt=".2f",                     # format numbers with 2 decimals
            cmap="coolwarm",               # colour scheme: blue (negative) to red (positive)
            center=0)                      # put 0 at the middle colour
plt.title("Feature Correlation Matrix")
plt.tight_layout()                         # tidy spacing so labels are not cut off
plt.show()

# ============================================================================
#== END OF Part 3 SETUP CODE =================================================

# X = the input columns. We pass the LIST of feature names inside [ ] to pick those columns.
X = df[feature_names]

# y = the single column we want to predict.
y = df["Heating_Load"]

# ---- Split into training and test sets ------------------------------------
# We hide 20% of the buildings as a "test set" the model never sees during training.
# This lets us check honestly how well it predicts NEW, unseen buildings.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,                       # the data to split
    test_size=0.20,             # 0.20 = keep 20% for testing, 80% for training
    random_state=RANDOM_STATE)  # fixed seed so the split is the same for everyone

# ---- Scale the features ----------------------------------------------------
# Features have very different ranges (areas in hundreds, glazing between 0 and 0.4).
# Scaling rescales each feature to a comparable range, which helps many models.
scaler = StandardScaler()                   # create the scaler tool
X_train_s = scaler.fit_transform(X_train)   # LEARN the scaling from training data AND apply it
X_test_s  = scaler.transform(X_test)        # apply the SAME scaling to the test data
# Important: we learn scaling from training data only, so no test information "leaks" in.

print("----------------------")
print("----------------------")
print(f"Training set: {X_train.shape[0]} buildings")
print(f"Test set:     {X_test.shape[0]} buildings")

# ============================================================================
#== END OF Part 4 SETUP CODE =================================================

# ---- Model 1: Linear Regression -------------------------------------------
lin = LinearRegression()             # create the (empty, untrained) model
lin.fit(X_train_s, y_train)          # .fit(...) = TRAIN it on the training data
y_pred_lin = lin.predict(X_test_s)   # .predict(...) = make predictions for the test buildings

# ---- Model 2: Random Forest ------------------------------------------------
rf = RandomForestRegressor(
        n_estimators=50,            # build 200 decision trees and average them
        random_state=RANDOM_STATE)   # fixed seed for reproducibility
rf.fit(X_train_s, y_train)
y_pred_rf = rf.predict(X_test_s)

# ---- A small helper function to print all three scores neatly --------------
def report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))   # np.sqrt = square root
    r2   = r2_score(y_true, y_pred)
    # The :5.2f below means "show as a number, width 5, with 2 decimals".
    print(f"{name:18s} | MAE={mae:5.2f} | RMSE={rmse:5.2f} | R²={r2:5.3f}")

print("Performance on the TEST set (lower error / higher R² is better):")
report("Linear Regression", y_test, y_pred_lin)
report("Random Forest",     y_test, y_pred_rf)

# ---- A picture is clearer than numbers: predicted vs actual ----------------
# Each dot is one test building. The closer the dots sit to the red diagonal
# line, the closer the prediction was to the true value.
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.6, color="#2F5496", label="Random Forest")
#   alpha=0.6 makes dots slightly see-through so overlaps are visible.
lims = [y_test.min(), y_test.max()]                 # the smallest and largest true values
plt.plot(lims, lims, "r--", label="Perfect prediction")   # red dashed diagonal line
plt.xlabel("Actual Heating Load"); plt.ylabel("Predicted Heating Load")
plt.title("Random Forest — Predicted vs Actual")
plt.legend()                                        # show the small key/legend box
plt.show()


# ---- Which inputs matter most? (Feature importance) ------------------------
# A Random Forest can tell us how useful each feature was for its predictions.
# This is valuable insight: it shows facility teams where to focus.
importance = pd.Series(rf.feature_importances_,     # the importance score of each feature
                       index=feature_names           # label each score with its feature name
                       ).sort_values()               # sort from smallest to largest

plt.figure(figsize=(8, 5))
importance.plot(kind="barh", color="#2F5496")        # "barh" = horizontal bar chart
plt.title("Feature Importance for Heating Load")
plt.xlabel("Relative importance")
plt.tight_layout(); plt.show()

# Print the top 3 most important features in plain text.
print("Top 3 drivers of heating load:")
for feat, imp in importance.sort_values(ascending=False).head(3).items():
    # A "for" loop repeats the indented line once for each of the top 3 items.
    print(f"   • {feat.replace('_', ' ')}: {imp:.1%}")   # :.1% shows the value as a percentage

 
# ============================================================================
#== END OF Part 5 SETUP CODE =================================================   

# ---- Turn the numeric target into 3 categories -----------------------------
# pd.qcut splits the values into 3 equal-sized groups (by quantiles).
# Lower heating load = more efficient, so the first band is the best.
labels = pd.qcut(df["Heating_Load"], q=3,
                 labels=["A (efficient)", "B (medium)", "C (poor)"])
y_clf = labels                                   # this is our new categorical target

# Split again, this time using the categorical target.
# stratify=y_clf keeps the same proportion of A/B/C in both train and test sets.
Xtr, Xte, ytr, yte = train_test_split(
    X, y_clf, test_size=0.20, random_state=RANDOM_STATE, stratify=y_clf)
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

# ---- Train a Random Forest CLASSIFIER --------------------------------------
clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
clf.fit(Xtr_s, ytr)
y_pred_clf = clf.predict(Xte_s)

# accuracy_score = fraction of test buildings whose band we predicted correctly.
print(f"Classification accuracy: {accuracy_score(yte, y_pred_clf):.1%}\n")

# classification_report breaks performance down per band (precision / recall / F1).
print("Detailed report per band:")
print(classification_report(yte, y_pred_clf))

# ---- Confusion matrix: where does the model get confused? ------------------
# Rows = the true band, columns = the predicted band. Numbers on the diagonal
# are CORRECT predictions; off-diagonal numbers are mistakes.
cm = confusion_matrix(yte, y_pred_clf, labels=clf.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=clf.classes_, yticklabels=clf.classes_)
#   fmt="d" = show whole numbers; the labels put band names on the axes.
plt.xlabel("Predicted label"); plt.ylabel("True label")
plt.title("Confusion Matrix — Efficiency Classification")
plt.tight_layout(); plt.show()

# ============================================================================
#== END OF Part 6 SETUP CODE ================================================= 

# ---- Final summary printout ------------------------------------------------
print("="*70)
print("LESSON SUMMARY — Supervised Learning for Building Energy")
print("="*70)
print(f"Best regression R²:      {r2_score(y_test, y_pred_rf):.3f} (Random Forest)")
print(f"Classification accuracy: {accuracy_score(yte, y_pred_clf):.1%}")
print("\nSkills practised: loading data, EDA, train/test split, scaling,")
print("regression, classification, evaluation metrics, feature importance.")
print("="*70)