import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# --- Step 1: Load and Inspect Data ---
# Loading the Iris dataset
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['species'] = pd.Categorical.from_codes(data.target, data.target_names)

print("Data Inspection (First 5 rows):")
print(df.head())
print("\nDataset Description:")
print(df.describe())

# --- Step 2 & 4: Visualizations (AI Improved) ---
# We are using Seaborn to create a clear, professional layout

# Set a clean visual style
sns.set_theme(style="whitegrid")

# Create a figure with 2 subplots: one for histograms, one for the correlation scatter
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Histograms of Petal Length (Distribution)
sns.histplot(data=df, x='petal length (cm)', hue='species', kde=True, ax=axes[0], palette='muted')
axes[0].set_title('Distribution of Petal Length', fontsize=14)

# Plot 2: Correlation Scatter Plot with Regression Line
# This visualizes the relationship between Petal Length and Petal Width
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)', hue='species', ax=axes[1], palette='muted', s=100)
# Adding a regression line (AI improvement suggestion) to show the trend clearly
sns.regplot(data=df, x='petal length (cm)', y='petal width (cm)', scatter=False, color='gray', ax=axes[1])

axes[1].set_title('Correlation: Petal Length vs Petal Width', fontsize=14)

# --- Step 3: Save the Figure ---
plt.tight_layout()
plt.savefig('correlation_plot.png', dpi=300)
print("\nAnalysis complete. Figure saved as 'correlation_plot.png'.")

# Show the plot
plt.show()
