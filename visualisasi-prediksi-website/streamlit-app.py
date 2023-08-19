import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import time

st.set_page_config(layout="wide")

# import data
df = pd.read_csv('final-EDA-jabodetabek-house-price.csv')
X = df.drop(['price_in_rp', 'district'], axis=1)
y = df['price_in_rp']


# title
st.header("""ISFEST 2023 : Final Data Competition - BEBAS""")

# layout
col1, col2 = st.columns([1,1], gap="large")
with col1:
    st.header("Model & Feature Selection")
    option = st.selectbox(
        'Choose Model',
        ('Select Model','Linear Regression', 'Ridge Regression', 'Lasso Regression', 'KNN', 'Desicion Tree', 'Random Forest')
    )
    # selected_columns = ['district', 'city', 'facilities', 'certificate', 'property_condition', 'building_size_m2', 'land_size_m2', 'maid_bathrooms','electricity', 'bedrooms','floors','carports','garages']
    city = st.selectbox(
        'City', 
        ('Bekasi','Bogor', 'Tangerang', 'Depok', 'Jakarta Selatan', 'Jakarta Barat', 'Jakarta Utara', 'Jakarta Timur', 'Jakarta Pusat')
    )
    building_size_m2 = st.number_input('Building Size (m2)', format='%f',value=272.0)
    certificate = st.selectbox(
        'Certificate',
        ('shm - sertifikat hak milik', 'hgb - hak guna bangunan', 'lainnya')
    )
    property_condition = st.selectbox(
        'Property Condition',
        ('bagus', 'baru', 'bagus sekali', 'sudah renovasi', 'butuh renovasi')
    )
    land_size_m2 = st.number_input('Land Size (m2)', format='%f', value=239.0)
    electricity = st.number_input('Electricity (mah)', format='%f', value=4400.0)
    bedrooms = float(st.number_input('Bedrooms', format='%d', value=2))
    
    # preprocessing input
    citymap = {
        'Bogor': 1.0,
        'Tangerang': 8.0,
        'Bekasi': 0.0,
        'Depok': 2.0,
        'Jakarta Selatan': 5.0,
        'Jakarta Barat': 3.0,
        'Jakarta Utara': 7.0,
        'Jakarta Timur': 6.0,
        'Jakarta Pusat': 4.0
    }
    city = citymap[city]

    building_size_m2 = np.sqrt(building_size_m2)

    certificate_map = {
        'shm - sertifikat hak milik': 2.0,
        'hgb - hak guna bangunan': 0.0,
        'lainnya': 1.0
    }
    certificate = certificate_map[certificate]

    property_condition_map = {
        'bagus': 0.0,
        'baru': 2.0,
        'bagus sekali': 1.0,
        'sudah renovasi': 4.0,
        'butuh renovasi': 3.0
    }
    property_condition = property_condition_map[property_condition]
    
    land_size_m2 = np.sqrt(land_size_m2)



with col2:
    st.header("Prediction")
    try:
        if option == 'Linear Regression':
            st.write('You selected:', option)
            model = LinearRegression()
        elif option == 'Ridge Regression':
            st.write('You selected:', option)
            model = Ridge(alpha=10)
        elif option == 'Lasso Regression':
            st.write('You selected:', option)
            model = Lasso(alpha=0.001)
        elif option == 'KNN':
            st.write('You selected:', option)
            model = KNeighborsRegressor(n_neighbors=3)
        elif option == 'Desicion Tree':
            st.write('You selected:', option)
            model = DecisionTreeRegressor(max_depth=8)
        elif option == 'Random Forest':
            st.write('You selected:', option)
            model = RandomForestRegressor(n_estimators=100,
                                    random_state=3,
                                    max_samples=0.5,
                                    max_features=0.75,
                                    max_depth=15)
        # Splitting data
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        # Melatih model pada set pelatihan
        # print(y)
        # print('test')
        with st.spinner('Wait for it...'):
            model.fit(X, y)

            X_test = pd.DataFrame({
                'city': [city],
                'building_size_m2': [building_size_m2],
                'certificate': [certificate],
                'property_condition': [property_condition],
                'land_size_m2': [land_size_m2],
                'electricity': [electricity],
                'bedrooms': [bedrooms]
            })

            y_pred = model.predict(X_test)
            

            try:
                st.write('Predicted Price:', y_pred[0][0])
            except:
                st.write('Predicted Price:', y_pred[0])
    except:
        st.write('Please select a model.')