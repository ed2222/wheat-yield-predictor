import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from get_temperatures import get_temperatures


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


csv_file_path = 'temp_yield.csv'


@st.cache_data
def load_data():
    return pd.read_csv(csv_file_path)

df = load_data()

all_countries = df['Country'].unique()

X = df[['avg_temp', 'HWN', 'HWM', 'CWN', 'CWM', 'Country','year']]
y = df['Yield_Value']
y_pred_all = model.predict(X)

df_predictions = pd.DataFrame({
    'year': df['year'],
    'Actual_Yield': df['Yield_Value'],
    'Predicted_Yield': y_pred_all
})


def main():
    st.title('Wheat Yield Predictor')  
    st.write('This app allows you to make predictions on wheat yield for a specific year and country.  \n',
             'The model uses a gradient boosting method and takes a year, a country, and several temperature-related parameters (see below) as features.')
    
    st.write('Temperature data are gathered from the [open-meteo climate API](https://open-meteo.com/en/docs/climate-api)  \n',
             'Model used: EC_Earth3P_HR (EC-Earth consortium, Rossby Center, Swedish Meteorological and Hydrological Institute/SMHI, Norrköping, Sweden)')  
    st.subheader('A look at the model performances:')  
    st.write('Here you can look at the model’s overall wheat yield prediction for every country that',
             'produces wheat. You can choose to look at the sum of all yields or at their mean.')

    agg_choice = st.selectbox("Choose aggregation type:", ["Sum", "Mean"])

    
    agg_method = 'sum' if agg_choice == 'Sum' else 'mean'

    
    df_grouped_predictions = df_predictions.groupby('year').agg({
        'Actual_Yield': agg_method,
        'Predicted_Yield': agg_method
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_grouped_predictions['year'], df_grouped_predictions['Actual_Yield']/1000, label='Actual Yield', color='blue', marker='o', linestyle='-', alpha=0.6)
    plt.plot(df_grouped_predictions['year'], df_grouped_predictions['Predicted_Yield']/1000, label='Predicted Yield', color='red', linestyle='--', alpha=0.6)
    plt.xlabel('Year')
    plt.ylabel('Yield Value (kg/ha)')
    plt.title(f'{agg_choice} over all countries of actual vs. predicted yield values by year')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt.gcf())
    plt.close()

    st.write('Here you can select any countries you want for a more thorough look.')  

    target_countries = st.multiselect(
            'Select Countries',
            options=all_countries,
            default=[]  
            )
    
    if target_countries:
        df_filtered_countries = df[df['Country'].isin(target_countries)]

        X_filtered = df_filtered_countries[['avg_temp', 'HWN', 'HWM', 'CWN', 'CWM', 'Country', 'year']]
        y_pred_filtered = model.predict(X_filtered)

        df_filtered_countries['Predicted_Yield'] = y_pred_filtered

        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab10(range(len(target_countries)))  

        
        for i, country in enumerate(target_countries):
            country_data = df_filtered_countries[df_filtered_countries['Country'] == country]
            plt.plot(country_data['year'], country_data['Yield_Value']/1000, color=colors[i], linestyle='-', alpha=0.8, label=f'{country} Actual Yield')  
            plt.plot(country_data['year'], country_data['Predicted_Yield']/1000, color=colors[i], linestyle='--', alpha=0.8, label=f'{country} Predicted Yield')  

        
        handles = []
        for i, country in enumerate(target_countries):
            handles.append(plt.Line2D([0], [0], color=colors[i], linestyle='-', label=f'{country}'))

        handles.extend([
            plt.Line2D([0], [0], color='black', linestyle='-', label='Actual Yield'),
            plt.Line2D([0], [0], color='black', linestyle='--', label='Predicted Yield')
        ])

        plt.xlabel('Year')
        plt.ylabel('Yield Value (kg/ha)')
        plt.title('Actual vs. Predicted Yield Value by Year for Selected Countries')  
        plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        st.pyplot(plt)
        plt.close()
    else:
        st.write("Please select at least one country.")

    st.markdown("""
                If you are interested, here are the metrics of the model:  
                - **Best Parameters:** learning_rate: 0.1, max_depth: 6, n_estimators: 400, subsample: 0.7
                - **Training R²:** 0.9821
                - **Test R²:** 0.9351  # Added colon
                - **Training MSE:** 5964117.1834
                - **Test MSE:** 20024000.3955
                - **Training MAPE:** 0.1119
                - **Test MAPE:** 0.1770  \n
                Note that the model you are looking at right now has been trained with the entire dataset, so its performance is slightly better.  
                """)
    
    st.subheader('Make Your Own Predictions')  
    st.write('If you want to use the model to make your own predictions, you will need values for the temperature-related features.',
             "But don't worry if you don't have those; we can provide you with several alternatives:")  
    st.markdown("""
                - If you don't have any data at all to provide, check the 'Use open-meteo climate API predictions' section.
                - If you have at least an average temperature over the wheat growth period (see below) for each year and country, check the 'Custom Prediction' section.
                """)
    
    st.markdown('##### What You Need to Know to Understand and Work with Wheat-Field-Predictor:')  

    st.markdown("""
                Methods and definitions adapted from:
                - [On the Measurement of Heat Waves](https://journals.ametsoc.org/view/journals/clim/26/13/jcli-d-12-00383.1.xml)
                - [Increasing Trends in Regional Heatwaves](https://www.nature.com/articles/s41467-020-16970-7)  

                Definitions needed to understand what follows:
                - **Growth Period:** Period from September of the previous year to June of the current year.  # Added colon
                - **Tmax:** 90th percentile of the average maximum temperatures over a window of 7 days for all data from 1950 to 1989 (40 years).  
                - **Tmin:** 10th percentile of the average minimum temperatures over a window of 7 days for all data from 1950 to 1989 (40 years).  
                - **Avg_temp:** Average of mean temperatures over the 'growth period'.
                - **Heat Wave:** Period of at least 3 days during which the maximum temperature is above Tmax.
                - **Cold Wave:** Period of at least 3 days during which the minimum temperature is below Tmin.
                - **Heat/Cold Wave Magnitude:** Difference between the average of maximum/minimum temperatures over the heat/cold wave period and Tmax/Tmin (positive for heat wave, negative for cold wave).
                - **HWN:** Heat Wave Number - number of 'heat waves' over the 'growth period'.
                - **HWM:** Heat Wave Magnitude - average magnitude of the 'heat waves' over the 'growth period'.
                - **CWN:** Cold Wave Number - number of 'cold waves' over the 'growth period'.
                - **CWM:** Cold Wave Magnitude - average magnitude of the 'cold waves' over the 'growth period'.
                """)
    
    st.markdown("""#### Use the Open-Meteo Climate API Predictions""")  
    st.write('Here you can use the Open-Meteo climate API predictions for future temperatures')  
    selected_country = st.selectbox('Select Country', options=all_countries)
    start_date = st.date_input('Start Date', value=pd.to_datetime('1961-01-01'), min_value=pd.to_datetime('1961-01-01'), max_value=pd.to_datetime('2050-12-31'))
    end_date = st.date_input('End Date', value=pd.to_datetime('2022-12-31'), min_value=pd.to_datetime('1961-01-01'), max_value=pd.to_datetime('2050-12-31'))

    temp_data = get_temperatures(country = selected_country, start = start_date, end =end_date)
    temp_data.rename(columns={'country': 'Country'}, inplace=True)

    om_X = temp_data[['avg_temp', 'HWN', 'HWM', 'CWN', 'CWM', 'Country', 'year']]

    if st.button('Get Predictions'):
        temp_data = get_temperatures(country=selected_country, start=start_date, end=end_date)
        temp_data.rename(columns={'country': 'Country'}, inplace=True)

        om_X = temp_data[['avg_temp', 'HWN', 'HWM', 'CWN', 'CWM', 'Country', 'year']]

        if not om_X.empty:
            om_X['Predicted_Yield'] = model.predict(om_X)
            st.write(f'Predictions for {selected_country} from {start_date} to {end_date}:')

            plt.figure(figsize=(12, 8))
            plt.plot(om_X['year'], om_X['Predicted_Yield']/1000, label='Predicted Yield', color='red', linestyle='--', alpha=0.6)
            plt.xlabel('Year')
            plt.ylabel('Yield Value (kg/ha)')
            plt.title(f'Wheat yield prediction for {selected_country} using open-meteo climate predictions')
            plt.grid(True)
            st.pyplot(plt)
            plt.close()
        else:
            st.write('No data available for the selected country and date range.')

    st.markdown("""#### Custom predictions""")
    st.write('Here you can make predictions using custom values for temperature paramaters. You are requiered to have a list value to give',
             "for `avg_temp`. Then choose between the 3 following scenari :")
    st.markdown("""
                - **smooth scenario** : values of CWN, CWM, HWN and HWM are set to 0
                - **average scenario** : values of CWN, CWM, HWN and HWM are set to their average
                - **extrem scenario** : values of CWN, CWM, HWN and HWM are set to the maximal reported up until today 
                - **full custom scenario** : enter CWN, CWM, HWN and HWM values yourself
                """)
    
    
    avg_temp = st.text_input('Average Temperature (comma-separated values)')
    country = st.selectbox('Select a Country', options=all_countries)
    year = st.number_input(' Starting year', min_value=1961, max_value=2050, step=1)

    st.write(year)
    scenario = st.selectbox('Select Scenario:', ['Smooth', 'Average', 'Extreme', 'Full Custom'])

   
    if scenario == 'Full Custom':
        HWN = st.number_input('Heat Wave Number (HWN)', min_value=0, step=1)
        HWM = st.number_input('Heat Wave Magnitude (HWM)', min_value=0.0, step=0.1)
        CWN = st.number_input('Cold Wave Number (CWN)', min_value=0, step=1)
        CWM = st.number_input('Cold Wave Magnitude (CWM)', min_value=0.0, step=0.1)
    else:
        HWN = df['HWN'].mean() if scenario == 'Average' else df['HWN'].max() if scenario == 'Extreme' else 0
        HWM = df['HWM'].mean() if scenario == 'Average' else df['HWM'].max() if scenario == 'Extreme' else 0
        CWN = df['CWN'].mean() if scenario == 'Average' else df['CWN'].max() if scenario == 'Extreme' else 0
        CWM = df['CWM'].mean() if scenario == 'Average' else df['CWM'].max() if scenario == 'Extreme' else 0

    
    if st.button('Generate Prediction'):
        if avg_temp:
            avg_temp_values = list(map(float, avg_temp.split(',')))

            
            input_data = pd.DataFrame({
                'avg_temp': avg_temp_values,
                'HWN': [HWN] * len(avg_temp_values),
                'HWM': [HWM] * len(avg_temp_values),
                'CWN': [CWN] * len(avg_temp_values),
                'CWM': [CWM] * len(avg_temp_values),
                'Country': [country] * len(avg_temp_values),
                'year': range(year, year + len(avg_temp_values))
            })

            predictions = model.predict(input_data)
            st.write(f'Predictions for {country} in {year} using the "{scenario}" scenario:')
            st.write(predictions)

            plt.figure(figsize=(12, 6))
            plt.plot(range(year, year + len(avg_temp_values)), predictions/1000, label='Predicted Yield', color='red', linestyle='--', alpha=0.6)
            plt.xlabel('Year')
            plt.xlim(year - 1, year + len(avg_temp_values) + 1)
            plt.ylabel('Yield Value (kg/ha)')
            plt.title(f'Custom Prediction for {country} in {year} using the "{scenario}" scenario')
            plt.legend()
            plt.grid(True)
            st.pyplot(plt)
            plt.close()
        else:
            st.write('Please provide values for average temperature.')


if __name__ == '__main__':
    main()