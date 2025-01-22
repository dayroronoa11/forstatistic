from navigation import make_sidebar, make_filter
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from data_processing import finalize_data

# Initialize sidebar and fetch data
make_sidebar()

# Fetch survey data, credentials, and combined data
df_survey, df_creds, combined_df = finalize_data()

# Streamlit UI components
st.title('Importance-Performance Analysis for Employees')

# Display DataFrame
st.write("Importance-Performance Analysis Table:")
columns_list = [
    'unit', 'subunit', 'directorate', 'division', 'department', 'section',
    'layer', 'status', 'generation', 'gender', 'marital', 'education',
    'tenure_category', 'children', 'region', 'participation_23'
]
filtered_data, selected_filters = make_filter(columns_list, df_survey)

# Prefix Mapping for independent variable categories
prefix_mapping = {
    "SAT": "Overall Satisfaction",
    "KD": "Kebutuhan Dasar",
    "KI": "Kontribusi Individu",
    "KR": "Kerjasama",
    "PR": "Pertumbuhan",
    "TU": "Tujuan",
    "KE": "Keterlekatan"
}

# List of independent variables
independent_vars = ['KD1', 'KD2', 'KD3', 'KI1', 'KI2', 'KI3', 'KI4', 'KI5', 'KR1', 'KR2', 'KR3', 
                    'KR4', 'KR5', 'PR1', 'PR2', 'TU1', 'TU2', 'KE1', 'KE2', 'KE3']

# Mean of SAT (Performance) and mean of B variables (Importance)
performance_mean = filtered_data['SAT'].mean()
importance_mean = filtered_data[independent_vars].mean()

# Standardize the independent variables and fit linear regression for Importance (X) based on SAT (Y)
scaler = StandardScaler()
X = scaler.fit_transform(filtered_data[independent_vars])  # Use filtered_data here
y = scaler.fit_transform(filtered_data[['SAT']]).flatten()  # Use filtered_data here

model = LinearRegression()
model.fit(X, y)

# Using the Standardized Beta (St B) coefficients for Importance (not absolute values)
importance_values = model.coef_  # Standardized beta coefficients as importance

# Construct the correlation_df DataFrame with Standardized Beta (St B) values
correlation_df = pd.DataFrame({
    'Factor': independent_vars,
    'Importance': [round(beta, 3) for beta in importance_values],  # Directly using St B as importance
    'Performance': [round(beta, 3) for beta in importance_mean.values]
})

# Midpoint thresholds for dynamic quadrants
importance_min, importance_max = correlation_df['Importance'].min(), correlation_df['Importance'].max()
performance_min, performance_max = correlation_df['Performance'].min(), correlation_df['Performance'].max()

importance_midpoint = (importance_max + importance_min) / 2
performance_midpoint = (performance_max + performance_min) / 2

# Function to classify factors into quadrants based on midpoints
def classify_factor_dynamic(importance, performance, importance_midpoint, performance_midpoint):
    if importance > importance_midpoint and performance > performance_midpoint:
        return 'High Importance, High Performance (Keep doing well)'
    elif importance > importance_midpoint and performance <= performance_midpoint:
        return 'High Importance, Low Performance (Improve performance)'
    elif importance <= importance_midpoint and performance > performance_midpoint:
        return 'Low Importance, High Performance (Possible overkill)'
    else:
        return 'Low Importance, Low Performance (Low priority)'

# Classify each factor using the midpoint thresholds
correlation_df['Category'] = [
    classify_factor_dynamic(row['Importance'], row['Performance'], 
                            importance_midpoint, performance_midpoint)
    for _, row in correlation_df.iterrows()
]

st.dataframe(correlation_df)

# Scatter plot for Importance-Performance Analysis
fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot points with color map
scatter = ax.scatter(correlation_df['Performance'], correlation_df['Importance'], 
                     c=correlation_df['Category'].astype('category').cat.codes, 
                     cmap='viridis', s=100, alpha=0.7)

# Label each factor with better placement
for i, row in correlation_df.iterrows():
    ax.text(row['Performance'] + 0.01, row['Importance'], row['Factor'], 
            fontsize=9, ha='left', va='bottom')

# Adjust limits to center the plot around the midpoints
#ax.set_xlim(min(performance_min, performance_midpoint - 0.4), max(performance_max, performance_midpoint + 0.4))
#ax.set_ylim(min(importance_min, importance_midpoint - 0.2), max(importance_max, importance_midpoint + 0.2))

# Add dynamic quadrant lines
ax.axhline(y=importance_midpoint, color='green', linestyle='--', label="Importance Midpoint")
ax.axvline(x=performance_midpoint, color='red', linestyle='--', label="Performance Midpoint")

# Set axis labels and title
ax.set_xlabel('Performance (Mean of SAT)', fontsize=12, labelpad=10)
ax.set_ylabel('Importance (Standardized Beta)', fontsize=12, labelpad=10)
ax.set_title('Importance-Performance Analysis', fontsize=16, pad=20)

# Add grid lines and adjust legend
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='lower right', bbox_to_anchor=(1, 0), fontsize=10, markerscale=1.5)

# Format tick labels to 2 decimal places
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# Add text annotations for each quadrant
# Top-left quadrant
#ax.text(performance_min, importance_max, 'Concentrate Here', 
        #ha='center', va='center', fontsize=10, color='blue', fontweight='bold')

# Top-right quadrant
#ax.text(performance_max - 0.05, importance_max - 0.05, 'Keep up the good work', 
        #ha='center', va='center', fontsize=10, color='purple', fontweight='bold')

# Bottom-left quadrant
#ax.text(performance_min + 0.05, importance_min + 0.05, 'Low priority', 
        #ha='center', va='center', fontsize=10, color='red', fontweight='bold')

# Bottom-right quadrant
#ax.text(performance_max - 0.05, importance_min + 0.05, 'Possible overkill', 
        #ha='center', va='center', fontsize=10, color='green', fontweight='bold')

# Display plot
st.pyplot(fig)

# Classification of Independent Variables
st.write("Classification of Independent Variables:")
for category in correlation_df['Category'].unique():
    with st.expander(f"{category}"):

        factors_in_category = correlation_df[correlation_df['Category'] == category]['Factor']
        st.write(", ".join(factors_in_category))



