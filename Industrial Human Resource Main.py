import pandas as pd
import numpy as np
import plotly.express as plt
import streamlit as st
import json
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import folium
import nltk
nltk.download('punkt')
import seaborn as sns

df=pd.read_csv("C:\\Users\\Khiza\\Downloads\\DataSets-20240214T190749Z-001\\IndustrialHumanResource.csv", encoding='cp1252')

# Load the geojson file
geojson_url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
response = urllib.request.urlopen(geojson_url)
ta = json.loads(response.read())
#Data Cleaning
df["ï»¿State Code"]=df["ï»¿State Code"].str.lstrip("`")
df["District Code"]=df["District Code"].str.lstrip("`")
df["Division"]=df["Division"].str.lstrip("`")
df["Group"]=df["Group"].str.lstrip("`")
df["Class"]=df["Class"].str.lstrip("`")

# Data Exploration
st.title('Industrial Classification of Main and Marginal Workers')
st.write('This app visualizes the workers population of various industries with respect to various geographies.')

# Filter the data by state
state_options = df['India/States'].unique().tolist()
selected_state = st.selectbox('Select a state', state_options)

# Filter the data by division
division_options = df[df['India/States'] == selected_state]['Division'].unique().tolist()
selected_division = st.selectbox('Select a division', division_options)

# Filter the data by group
group_options = df[(df['India/States'] == selected_state)]['Group'].unique().tolist()
selected_group = st.selectbox('Select a group', group_options)

# Filter the data by class
class_options = df[(df['India/States'] == selected_state) & (df['Group'] == selected_group)]['Class'].unique().tolist()
selected_class = st.selectbox('Select a class', class_options)

# Filter the data by nic name
nic_name_options = df[(df['India/States'] == selected_state) & (df['Group'] == selected_group) & (df['Class'] == selected_class)]['NIC Name'].unique().tolist()
selected_nic_name = st.selectbox('Select a NIC Name', nic_name_options)

# Filter the data by work type (main or marginal)
work_type_options = ['Main Workers', 'Marginal Workers']
selected_work_type = st.selectbox('Select a work type', work_type_options)

# Filter the data by gender (males or females)
gender_options = ['Males', 'Females']
selected_gender = st.selectbox('Select a gender', gender_options)

# Filter the data by location (rural or urban)
location_options = ['Rural', 'Urban']
selected_location = st.selectbox('Select a location', location_options)

df['total_workers'] = df['Main Workers - Total -  Persons'] + df['Marginal Workers - Total -  Persons']
df['total_workers'] = df['total_workers'].astype(str)

filtered_data = df[(df['India/States'] == selected_state) & (df['Division'] == selected_division) & (df['Group'] == selected_group) & (df['Class'] == selected_class) & (df['NIC Name'] == selected_nic_name)]

if selected_work_type != 'All':
    filtered_data = filtered_data[filtered_data['total_workers'].str.contains(selected_work_type)]

if selected_gender != 'All':
    filtered_data = filtered_data[filtered_data['total_workers'].str.contains(selected_gender)]

if selected_location != 'All':
    filtered_data = filtered_data[filtered_data['total_workers'].str.contains(selected_location)]

# Modify the India/States column to match the keys in the geojson file
filtered_data['India/States'] = filtered_data['India/States'].replace({'Andhra Pradesh': 'Andhra Pradesh',
                                                                      'Arunachal Pradesh': 'Arunachal Pradesh',
                                                                      'Assam': 'Assam',
                                                                      'Bihar': 'Bihar',
                                                                      'Chhattisgarh': 'Chhattisgarh',
                                                                      'Goa': 'Goa',
                                                                      'Gujarat': 'Gujarat',
                                                                      'Haryana': 'Haryana',
                                                                      'Himachal Pradesh': 'Himachal Pradesh',
                                                                      'Jharkhand': 'Jharkhand',
                                                                      'Karnataka': 'Karnataka',
                                                                      'Kerala': 'Kerala',
                                                                      'Madhya Pradesh': 'Madhya Pradesh',
                                                                      'Maharashtra': 'Maharashtra',
                                                                      'Manipur': 'Manipur',
                                                                      'Meghalaya': 'Meghalaya',
                                                                      'Mizoram': 'Mizoram',
                                                                      'Nagaland': 'Nagaland',
                                                                      'Odisha': 'Odisha',
                                                                      'Punjab': 'Punjab',
                                                                      'Rajasthan': 'Rajasthan',
                                                                      'Sikkim': 'Sikkim',
                                                                      'Tamil Nadu': 'Tamil Nadu',
                                                                      'Telangana': 'Telangana',
                                                                      'Tripura': 'Tripura',
                                                                      'Uttar Pradesh': 'Uttar Pradesh',
                                                                      'Uttarakhand': 'Uttarakhand',
                                                                      'West Bengal': 'West Bengal'})




# Data Cleaning
st.subheader('Data Cleaning')
st.write('No data cleaning is required for this dataset.')

# Feature Engineering
st.subheader('Feature Engineering')
from nltk.tokenize import word_tokenize, sent_tokenize

# tokenize sentences 
sentences = df['NIC Name'].apply(sent_tokenize)

# Display the sentences
st.write(sentences)

# Model Building and Testing
st.subheader('Model Building and Testing')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Select features and target variable
# Replace 'Feature1', 'Feature2', 'Feature3' with actual column names from your DataFrame
features = ['Main Workers - Total -  Persons', 'Marginal Workers - Total -  Persons']  # Example features
target = 'total_workers'

# Check if selected features and target variable exist in the DataFrame
if all(feature in df.columns for feature in features) and target in df.columns:
    X = df[features]
    y = df[target].astype(int)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.write('Mean Squared Error:', mse)
    st.write('R-squared Score:', r2)
    st.write(model.intercept_, model.coef_)   
    
    # Create a scatter plot to visualize the predicted vs actual values
    scatter_data = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    scatter_fig = plt.scatter(scatter_data, x='Actual', y='Predicted', title='Predicted vs Actual', trendline='ols')
    st.plotly_chart(scatter_fig, use_container_width=True)
    
else:
    st.write('Selected features or target variable not found in the DataFrame.')

# Visualize the data
fig = plt.choropleth(df, 
                    geojson=ta,  # Use the loaded GeoJSON data directly
                    locations='India/States', 
                    color='total_workers', 
                    hover_name='India/States', 
                    projection='mercator', 
                    title='Industrial Classification of Main and Marginal Workers')
fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig,use_container_width=True)
