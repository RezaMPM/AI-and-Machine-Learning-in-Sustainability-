# AI-and-Machine-Learning-in-Sustainability-
AI and Machine Learning in Sustainability  
===========================================
## Part 5 code explanation :   
This code snippet represents a classic machine learning workflow in Python (likely using the `scikit-learn` library). It sets up two different predictive models, trains them on a dataset, makes predictions, and then compares their accuracy.

Here is a step-by-step breakdown of what each section does:

### **Model 1: Linear Regression**

This section builds a simple baseline model that tries to draw a straight line through the data to make predictions.

* **`lin = LinearRegression()`**: This initializes a blank, untrained Linear Regression model.
* **`lin.fit(X_train_s, y_train)`**: The `.fit()` method is where the actual "learning" happens. The model is fed the training features (`X_train_s`) and the true target values (`y_train`) so it can figure out the mathematical relationship between them.
* **`y_pred_lin = lin.predict(X_test_s)`**: The `.predict()` method takes the unseen testing features (`X_test_s`) and asks the trained model to guess the target values. These guesses are saved in the variable `y_pred_lin`.

### **Model 2: Random Forest**

This section builds a more complex, powerful model. A Random Forest makes predictions by combining the results of many individual "decision trees."

* **`rf = RandomForestRegressor(...)`**: This creates the empty Random Forest model with two specific settings (hyperparameters):
* `n_estimators=200`: Tells the model to build 200 individual decision trees and average their predictions together.
* `random_state=RANDOM_STATE`: Ensures that the random processes used to build the trees happen exactly the same way every time you run the code, making your results reproducible.


* **`rf.fit(...)` and `rf.predict(...)**`: Exactly like the Linear Regression model, this trains the Random Forest on the training data and then uses it to guess the answers for the test data.

### **The Helper Function: `report**`

This custom function (`def report(...)`) is built to calculate and neatly print three standard performance metrics so you can compare the two models.

* **`mae = mean_absolute_error(...)`**: Calculates the **Mean Absolute Error**. This tells you, on average, how far off the model's predictions were from the actual values. Lower is better.
* **`rmse = np.sqrt(mean_squared_error(...))`**: Calculates the **Root Mean Squared Error**. Similar to MAE, but because it squares the errors before averaging them, it heavily penalizes models for making extremely large mistakes. Lower is better.
* **`r2 = r2_score(...)`**: Calculates the **R-squared score**. This measures how well the model explains the variance in the data. An R² of 1.0 is perfect, 0.0 is basically guessing the average every time, and negative numbers mean the model is worse than just guessing the average. Higher is better.
* **`print(f"{name:18s} | ...")`**: This uses Python formatted strings (f-strings) to print the results in an aligned table. For example, `mae:5.2f` tells Python to print the MAE taking up 5 characters of space, with exactly 2 decimal places.

### **The Execution**

The final three lines simply print a header row and then call the `report` function twice—once for the Linear Regression predictions (`y_pred_lin`) and once for the Random Forest predictions (`y_pred_rf`)—allowing you to easily see which model performed better on the test data.
