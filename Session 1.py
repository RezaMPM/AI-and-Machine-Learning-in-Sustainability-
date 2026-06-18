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