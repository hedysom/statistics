"""Python code submission file for assignment 2.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the .png images showing your plots for each question are generated.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

# ========================
# Task 1: MPG Dataset
# ========================

# Load dataset and drop missing values
mpg = sns.load_dataset("mpg").dropna()

# a) MPG dataset statistics
print("[MPG Dataset] Summary Statistics:")

# calculate and print the statistics of the mpg dataset
# Start


# Do simply describe
print(mpg.describe())
"""
# or use this for each column like mpg, cilinders etc.
mean = mpg["mpg"].mean()
median = mpg["mpg"].median()
min_val = mpg["mpg"].min()
max_val = mpg["mpg"].max()
std_dev = mpg["mpg"].std()
q1 = mpg["mpg"].quantile(0.25)
q2 = mpg["mpg"].quantile(0.50)
q3 = mpg["mpg"].quantile(0.75)


print("\nDescriptive Statistics for mpg:")
print("Mean:", round(mean, 2))
print("Median:", round(median, 2))
print("Min:", round(min_val, 2))
print("Max:", round(max_val, 2))
print("Std Dev:", round(std_dev, 2))
print("1st quartile (25%):", round(q1, 2))
print("2nd quartile (50%):", round(q2, 2))
print("3rd quartile (75%):", round(q3, 2))
"""
# Stop

# b) Scatter plot: Weight vs MPG
plt.figure(figsize=(8, 6))
sns.scatterplot(data=mpg, x="weight", y="mpg")
plt.title("Weight vs MPG")
plt.xlabel("Weight")
plt.ylabel("Miles per Gallon")
plt.grid(True)
plt.show()

# Pearson correlation coefficient

# calculate pearson correlation coefficient (use the stats.pearsonr function that returns r and p_value)
# Start

r, p_value = stats.pearsonr(mpg["weight"], mpg["mpg"])

# Stop
print(f"[MPG Dataset] Correlation coefficient (r): {r:.3f}")
print(f"[MPG Dataset] P-value: {p_value:.4f}")


# c) Fit regression line

# calculate the regression line (use the stats.linregress function that returns intercept and slope)
# Start
slope, intercept, r, p, se = stats.linregress(mpg["weight"], mpg["mpg"])
print(f"[MPG Dataset] Correlation coefficient (r): {r:.3f}")
print(f"[MPG Dataset] P-value: {p:.4f}")

# Stop

print(f"[MPG Dataset] Regression line: mpg = {intercept:.2f} + {slope:.5f} * weight")

# Overlay fitted regression line
plt.figure(figsize=(8, 6))
sns.scatterplot(data=mpg, x="weight", y="mpg", label="Data Points", alpha=0.6)

# write the expression for line = ?
# Start

line = intercept + slope * mpg["weight"]

# Stop


plt.plot(
    mpg["weight"],
    line,
    color="red",
    label=f"Fitted Line: mpg = {intercept:.2f} + {slope:.5f} * weight",
)
plt.title("Linear Fit: MPG vs Weight with Fitted Line")
plt.xlabel("Weight")
plt.ylabel("Miles per Gallon")
plt.legend()
plt.grid(True)
plt.show()

# Predict mpg for a car with weight = 3200 lbs
weight_pred = 3200

# write the expression for mpg_pred = ?
# Start

mpg_pred = intercept + slope * weight_pred

# Stop

print(f"[MPG Dataset] Predicted MPG for weight 3200 lbs: {mpg_pred:.2f}")

# d) Rule of thumb prediction
# calculate the std (s_y) and change (pred_change) of mpg
# Start

s_y = mpg["mpg"].std()
pred_change = r * s_y

# Stop

print(
    f"[MPG Dataset] Estimated change in mpg (rule of thumb): {
        pred_change:.2f
    } (for +1 std in weight)"
)

# Compact rule of thumb prediction

# calculate the following values
# Start
mean_weight = mpg["weight"].mean()
std_weight = mpg["weight"].std()
mean_mpg = mpg["mpg"].mean()
z_x = (3200 - mean_weight) / std_weight
# Stop
z_y = r * z_x
mpg_est = mean_mpg + z_y * s_y

print(f"[MPG Dataset] Compact rule of thumb estimate: {mpg_est:.2f}")

# ========================
# Task 2: Tips Dataset
# ========================


# Load the tips dataset
tips = sns.load_dataset("tips")

# a) Scatter plot: Total Bill vs Tip, Size vs Tip

plt.figure(figsize=(8, 6))
sns.scatterplot(data=tips, x="total_bill", y="tip")
plt.title("Scatter Plot: Total Bill vs Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(data=tips, x="size", y="tip")
plt.title("Scatter Plot: Size vs Tip")
plt.xlabel("Party Size")
plt.ylabel("Tip")
plt.grid(True)
plt.show()

# b) Pearson correlation between total_bill and tip

# calculate the correlation coefficient r using stats.pearsonr
# Start
r, p_value = stats.pearsonr(tips["total_bill"], tips["tip"])
r1, p_value1 = stats.pearsonr(tips["size"], tips["tip"])
# Stop

print(f"[Tips Dataset] Correlation coefficient (r) between Total Bill and Tip: {r:.3f}")

print(f"[Tips Dataset] Correlation coefficient (r) between Size and Tip: {r1:.3f}")

X = tips[["total_bill", "size"]]
y = tips["tip"]

model = LinearRegression()
model.fit(X, y)


# Start
intercept = model.intercept_
coefficients = model.coef_
# Stop

r_squared = model.score(X, y)

print(
    f"[Tips Dataset] Regression equation: tip = {intercept:.2f} + {
        coefficients[0]:.2f
    } * total_bill + {coefficients[1]:.2f} * size"
)
print(f"[Tips Dataset] R^2 value: {r_squared:.3f}")

# Predict tip for total_bill = 25 and size = 4

# Start
predicted_tip = model.predict([[25, 4]])
predicted_tip = predicted_tip[0]
# Stop
print(f"[Tips Dataset] Predicted tip for total_bill=25 and size=4: {predicted_tip:.2f}")


# c) 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    tips["total_bill"], tips["tip"], tips["size"], c="blue", marker="o", alpha=0.6
)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
ax.set_zlabel("Size")
ax.set_title("3D Scatter Plot: Total Bill, Tip, and Size")
plt.show()

# add regression plane
X_plane = np.array([tips["total_bill"], tips["size"]]).T
y_plane = tips["tip"]
reg = LinearRegression()
reg.fit(X_plane, y_plane)

total_bill_range = np.linspace(tips["total_bill"].min(), tips["total_bill"].max(), 10)
size_range = np.linspace(tips["size"].min(), tips["size"].max(), 10)
total_bill_grid, size_grid = np.meshgrid(total_bill_range, size_range)
tip_grid = reg.predict(np.c_[total_bill_grid.ravel(), size_grid.ravel()]).reshape(
    total_bill_grid.shape
)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(
    tips["total_bill"], tips["tip"], tips["size"], c="blue", marker="o", alpha=0.6
)
ax.plot_surface(total_bill_grid, size_grid, tip_grid, color="red", alpha=0.3)
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")
ax.set_zlabel("Size")
ax.set_title("3D Regression Plane: Total Bill, Tip, and Size")
plt.show()
