from navigation import make_sidebar, make_filter
import streamlit as st
import pandas as pd
import plotly.express as px
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
    st.write("##### Likelihood to Stay (LS)")
    st.write(
        "- **Loyal Enthusiast**: High SAT, High LS\n"
        "- **Contented Wanderers**: High SAT, Low LS\n"
        "- **Reluctant Stayers**: Low SAT, High LS\n"
        "- **Disengaged Flight to Risk**: Low SAT, Low LS"
    )
with col2:
    st.write("##### Net Promoter Score (NPS)")
    st.write(
        "- **Brand Champions**: High SAT, Promoter\n"
        "- **Satisfied Critics**: High SAT, Detractor\n"
        "- **Loyal Promoters**: Low SAT, Promoter\n"
        "- **Vocal Detractors**: Low SAT, Detractor"
    )

# Categorization Functions
def categorize_ls(row):
    if row['SAT'] >= 4 and row['KE1'] >= 4:
        return 'Loyal Enthusiast'
    elif row['SAT'] >= 4 and row['KE1'] <= 2:
        return 'Contented Wanderers'
    elif row['SAT'] <= 2 and row['KE1'] >= 4:
        return 'Reluctant Stayers'
    elif row['SAT'] <= 2 and row['KE1'] <= 2:
        return 'Disengaged Flight to Risk'
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
    return 'Neutral'

# Apply categorization to the original DataFrame
df_survey['LS_Category'] = df_survey.apply(categorize_ls, axis=1)
df_survey['NPS_Category'] = df_survey.apply(categorize_nps, axis=1)

# Apply filters to df_survey first
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

# Generate Figures Based on Filtered Data
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
