from navigation import make_sidebar, make_filter
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
from data_processing import finalize_data

# Streamlit UI setup
st.set_page_config(page_title='Combined IPA and Categorization', page_icon=':chart_with_upwards_trend:')
make_sidebar()
df_survey, df_creds, combined_df = finalize_data()

# Columns for potential filtering
columns_list = [
    'unit', 'subunit', 'directorate', 'division', 'department', 'section',
    'layer', 'status', 'generation', 'gender', 'marital', 'education',
    'tenure_category', 'children', 'region', 'participation_23'
]
st.subheader("IPA x Categorization", divider='grey')
col1, col2 = st.columns(2)
with col1: 
    st.write("##### Likehood to stay")
    st.write(
        "- **Loyal Enthusiast**: High SAT, High LS\n"
        "- **Contended Wanderers**: High SAT, Low LS\n"
        "- **Reluctant Stayers**: Low SAT, High LS\n"
        "- **Disengaged Flight to Risk**: Low SAT, Low LS"
    )
with col2 :
    st.write("##### NPS")
    st.write(
        "- **Brand Champions**: High SAT, Promoter\n"
        "- **Satisfied Critics**: High SAT, Detractor\n"
        "- **Loyal Promoters**: Low SAT, Promoter\n"
        "- **Vocal Detractors**: Low SAT, Detractor"
    )

# Assuming you have a DataFrame `df` with columns: 'SAT', 'LS', 'NPS'
def categorize_ls(row):
    if row['SAT'] >= 4 and row['KE1'] >= 4:
        return 'Loyal Enthusiast'
    elif row['SAT'] >= 4 and row['KE1'] <= 2:
        return 'Contented Wanderers'
    elif row['SAT'] <= 2 and row['KE1'] >= 4:
        return 'Reluctant Stayers'
    elif row['SAT'] <= 2 and row['KE1'] <= 2:
        return 'Disengaged Flight to Risk'
    elif row['SAT'] == 3 or row['KE1'] == 3:
        return 'Neutral'

def categorize_nps(row):
    if row['SAT'] >= 4 and row['NPS'] >= 9:
        return 'Brand Champions'
    elif row['SAT'] >= 4 and row['NPS'] <= 6:
        return 'Satisfied Critics'
    elif row['SAT'] <= 2 and row['NPS'] >= 9:
        return 'Loyal Promoters'
    elif row['SAT'] <= 2 and row['NPS'] <= 6:
        return 'Vocal Detractors'
    elif 7 <= row['NPS'] <= 8:
        return 'Neutral'

# Apply categorization
df_survey['LS_Category'] = df_survey.apply(categorize_ls, axis=1)
df_survey['NPS_Category'] = df_survey.apply(categorize_nps, axis=1)

filtered_data, selected_filters = make_filter(columns_list, df_survey)

# Add LS_Category and NPS_Category to filtered_data after filtering
if 'LS_Category' not in filtered_data.columns:
    filtered_data['LS_Category'] = filtered_data.apply(categorize_ls, axis=1)

if 'NPS_Category' not in filtered_data.columns:
    filtered_data['NPS_Category'] = filtered_data.apply(categorize_nps, axis=1)

# Handle null values
filtered_data['LS_Category'] = filtered_data['LS_Category'].fillna('Neutral')
filtered_data['NPS_Category'] = filtered_data['NPS_Category'].fillna('Neutral')

# Dropdown for LS categories
ls_filter = st.selectbox(
    "Filter by LS Category",
    options=["All", "Neutral", "Loyal Enthusiast", "Contented Wanderers", "Reluctant Stayers", "Disengaged Flight to Risk"]
)

# Dropdown for NPS categories
nps_filter = st.selectbox(
    "Filter by NPS Category",
    options=["All", "Neutral", "Brand Champions", "Satisfied Critics", "Loyal Promoters", "Vocal Detractors"]
)

# Apply additional filters to filtered_data based on LS and NPS
if ls_filter != "All":
    filtered_data = filtered_data[filtered_data['LS_Category'] == ls_filter]

if nps_filter != "All":
    filtered_data = filtered_data[filtered_data['NPS_Category'] == nps_filter]

# Display the filtered data
st.dataframe(filtered_data)

# LS Categories Bar Chart with Count and Percentage
ls_count = filtered_data['LS_Category'].value_counts().reset_index()
ls_count.columns = ['LS_Category', 'Count']
ls_count['Percentage'] = (ls_count['Count'] / ls_count['Count'].sum()) * 100

fig_ls = px.bar(
    ls_count, 
    x='LS_Category', 
    y='Count', 
    title='Distribution by LS Categories',
    text=ls_count.apply(lambda x: f"{x['Count']} ({x['Percentage']:.1f}%)", axis=1)
)
fig_ls.update_traces(textposition='outside')

# NPS Categories Bar Chart with Count and Percentage
nps_count = filtered_data['NPS_Category'].value_counts().reset_index()
nps_count.columns = ['NPS_Category', 'Count']
nps_count['Percentage'] = (nps_count['Count'] / nps_count['Count'].sum()) * 100

fig_nps = px.bar(
    nps_count,
    x='NPS_Category',
    y='Count',
    title='Distribution by NPS Categories',
    text=nps_count.apply(lambda x: f"{x['Count']} ({x['Percentage']:.1f}%)", axis=1)
)
fig_nps.update_traces(textposition='outside')

# Display Figures
st.plotly_chart(fig_ls)
st.plotly_chart(fig_nps)

st.divider()




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
performance_mean = df_survey['SAT'].mean()
importance_mean = df_survey[independent_vars].mean()

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



