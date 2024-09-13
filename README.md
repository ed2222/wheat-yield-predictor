# Wheat yield predictor

This is a web app that uses gradient boosting to predict wheat yield and gross production in any country.

## Steps
- Use meteorological data to reliably predict wheat yield and gross production in any country
- Extract relevant information from international crop statistics in order to clearly visualize important aspects of worldwide wheat production
- Create a web app to make this data available online and able to be freely manipulated by users to support custom predictions

For every country, a single set of GPS coordinates was used to get the temperature from the open-meteo climate API.
For most countries, the coordinates of the geographical center of the country (geocode/user agent: ‘google v3’) was used.
Exceptions : coordinates entered manually for major producers which have a large surface area (USA, Canada, etc). We used GPS coordinates of the most important wheat-producing regions in these countries.

## Files in /app repository

```Dockerfile``` : set of instructions required to build the Docker image

```app.py``` : source code for the streamlit app

```exceptions.txt``` : list of countries for which GPS coordinates have to be entered manually

```get_temperatures.py``` : Python file with the functions required to extract climate forecasting data from the openmeteo API

```heroku.yml``` : YAML file used by Heroku to define the deployment settings for the Docker app

```model.pkl``` : pickle file used for saving the machine learning model trained with sklearn

```requirements.txt``` : list of dependencies to be installed on the Docker image

```temp_yield.csv``` : worldwide wheat production and yield data extracted from a UN FAO dataset (https://www.fao.org/faostat/en/#data/QCL)

## Machine learning model
Model used: Gradient Boosting Regressor

Model metrics :
- Train R² : 0.9821
- Test R² : 0.9351
- Train MSE : 5964117.1834
- Test MSE : 20024000.3955
- Train MAPE : 0.1119
- Test MAPE : 0.1770

Variables used for model training :
- Country : cat. feature (productivity + wheat species)
- Year : num. feature (productivity)
- Average temperature : the average of mean temperatures (in °C) over a given growth period (period going from september of the previous year to june of the current year)
- Heat/cold wave[1][2] number : number of heat/cold waves over the growth period
- Heat/cold wave magnitude : average magnitude of heat/cold waves over the growth period

[1]https://journals.ametsoc.org/view/journals/clim/26/13/jcli-d-12-00383.1.xml

[2]https://www.nature.com/articles/s41467-020-16970-7


## Web app

Link to the openmeteo API : https://open-meteo.com/

Link to the Heroku app : https://wheat-yield-predictor-c05627a36d49.herokuapp.com/

Predictions can be made until 2050 using climate forecasting from the openmeteo API.
Users can provide their own climate predictions in the custom mode and choose among 4 scenari (or choose a fully customized prediction).
