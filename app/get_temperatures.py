###  You may need to run 
### ' pip install geopy ' 
### ' pip install openmeteo-requests ' 
###  and
### ' pip install requests-cache retry-requests numpy panda ' 
###  if not already installed


import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
import json
from geopy.geocoders import Nominatim
from retry_requests import retry


def get_data(lat, long, start, end) :

    ###  Data gathered from open-meteo climate API 
    ###  check 'https://open-meteo.com/en/docs/climate-api'
    ###  model used :  EC_Earth3P_HR 
    ###  (EC-Earth consortium, Rossby Center, Swedish Meteorological 
    ###   and Hydrological Institute/SMHI, Norrkoping, Sweden)

    
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

 
    url = "https://climate-api.open-meteo.com/v1/climate"
    params = {
        "latitude": lat,
        "longitude": long,
        "start_date": start,
        "end_date": end,
        "models": "EC_Earth3P_HR",
        "daily": ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"]
    }
    responses = openmeteo.weather_api(url, params=params)

    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min

    df= pd.DataFrame(data = daily_data)
    df2 = pd.DataFrame(data = daily_data)

    return df, df2


def rolling_percentile(series, percentile, window_size):

    result = []

    if window_size <= 0 :
        raise ValueError('window_size must be a strict positive integer')

    if window_size % 2 == 0:
        window_width = window_size/2

    else :
        window_width = (window_size-1)/2

    if window_width == 0:
        for i in range(len(series)):
            temps = series.iloc[i]
            result.append(np.percentile(temps, percentile))

    else :

        for i in range(len(series)):
            start = int(max(0, i - window_width))
            end = int(min(len(series), i + window_width))
            temps = series.iloc[start:end]
            flattened = [temp for sublist in temps for temp in sublist]
            result.append(np.percentile(flattened, percentile))

    return result


def identify_waves(df, temp_column, threshold_column, wave_type):

    waves = []
    current_wave = []

    for index, row in df.iterrows():
        if (wave_type == 'heat' and row[temp_column] > row[threshold_column]) or \
           (wave_type == 'cold' and row[temp_column] < row[threshold_column]):
            current_wave.append(row[temp_column] - row[threshold_column])

        else:
            if len(current_wave) >= 3:
                waves.append(current_wave)
            current_wave = []

    if len(current_wave) >= 3:
        waves.append(current_wave)

    return waves



def calculate_wave_metrics(df, period_mask, temp_column, threshold_column, wave_type):

    df_period = df[period_mask]
    waves = identify_waves(df_period, temp_column, threshold_column, wave_type)
    wave_count = len(waves)

    if wave_count > 0:
        wave_magnitude = sum([sum(wave)/len(wave) for wave in waves]) / wave_count

    else:
        wave_magnitude = 0

    return wave_count, wave_magnitude


def process_data(df, df2, start, end, window_size = 7) :


###  Methods and definitions adapted from :
###
###  - 'https://journals.ametsoc.org/view/journals/clim/26/13/jcli-d-12-00383.1.xml'
###  - 'https://www.nature.com/articles/s41467-020-16970-7'
###
###
###
###  Definitions needed to understand what follows :
###
###  - growth period : period going from september of the previous year to june of the curent year
###
###  - winter : period going from october of the previous year to february of the current year
###
###  - spring : period going from mars to june of the current year
###
###  - heat wave : period of at least 3 days over which the max temperature ('temperature_2m_max')
###                is above Tmax
###
###  - cold wave : period of at least 3 days over which the min temperature ('temperature_2m_min')
###                is under Tmin
###
###  - heat/cold wave magnitude : difference between the average of max/min temperature over heat/cold 
###                               wave period and Tmax/Tmin (positive for heat wave, negative for cold wave)
###
###
###
###
###
###   Process the data got on open-meteo API to return a dataframe that will contain the columns :
###
###   - Year : current year
###
###   - Tmax : 90th percentile of the average of the maximum of temperatures ('temperature_2m_max') 
###            over a window of size window size for all the data from 1950 to 1989 included (40 years)
###
###   - Tmin : 10th percentil of the average of the minimum of temperatures ('temperature_2m_min')
###            over a window of size window size for all the data from 1950 to 1989 included (40 years)
###
###   - avg_temp : average of mean temperatures ('temperature_2m_mean') over 'growth period'
###                (see definition above for 'growth period')
###
###   - avg_spring_temp : average of mean temperatures ('temperature_2m_mean') for the 'spring' perdiod 
###                       (see definition above for 'spring')
###
###   - avg_winter_temp : average of mean temperatures ('temperature_2m_mean') for the 'winter' perdiod
###                       (see definition above for 'winter')
###
###   - HWN : Heat Wave Number - number of 'heat waves' over the 'growth period'
###
###   - HWM : Heat Waves Magnitude - average magnitude of the 'heat wave(s)' over the 'growth period'
###
###   - CWN : Cold Wave Number - number of 'cold waves' over the 'growth period'
###
###   - CWM : Cold Waves Magnitude - average magnitude of the 'cold wave(s)' over the 'growth period'
###
###   - SHWN : Spring Heat Wave Number - number of 'heat waves' over the 'spring' period
###
###   - SHWM : Spring Heat Waves Magnitude - average magnitude of the 'heat wave(s)' over the 'spring' period
###
###   - WCWN : Winter Cold Wave Number - number of 'cold waves' over the 'winter' period
###
###   - WCWM : Winter Cold Waves Magnitude - average magnitude of the 'cold wave(s)' over the 'winter' period




    df['date'] = df['date'].dt.date
    df['date'] = df['date'].apply(lambda x : pd.Timestamp(x))

    df = df[(df['date'] >= pd.Timestamp(start)) & (df['date'] <= pd.Timestamp(end))]

    df['date'] = pd.to_datetime(df['date'])

    df['month_day'] = df['date'].dt.strftime('%m-%d')

    all_days = pd.date_range('1952-01-01', '1952-12-31').strftime('%m-%d')
    result_df = pd.DataFrame(all_days, columns=['date'])

    df['temperature_2m_max'] = df['temperature_2m_max'].round(2)
    df['temperature_2m_min'] = df['temperature_2m_min'].round(2)
    df['temperature_2m_mean'] = df['temperature_2m_mean'].round(2)


    grouped_max = df.groupby('month_day')['temperature_2m_max'].apply(list)
    grouped_min = df.groupby('month_day')['temperature_2m_min'].apply(list)


    sorted_max = grouped_max.apply(lambda x: sorted(x))
    sorted_min = grouped_min.apply(lambda x: sorted(x))


    result_df['temperatures_max'] = result_df['date'].map(sorted_max)
    result_df['temperatures_min'] = result_df['date'].map(sorted_min)

    result_df_shifted = pd.concat([result_df.iloc[-3:], result_df, result_df.iloc[:3]])
    result_df_shifted['temperatures_max'] = result_df_shifted['temperatures_max'].apply(lambda x: [round(t, 2) for t in x])
    result_df_shifted['temperatures_min'] = result_df_shifted['temperatures_min'].apply(lambda x: [round(t, 2) for t in x])

    result_df_shifted['Tmax'] = rolling_percentile(result_df_shifted['temperatures_max'], 90, 7)
    result_df_shifted['Tmin'] = rolling_percentile(result_df_shifted['temperatures_min'], 10, 7)

    final_result_df = result_df_shifted.iloc[3:-3].reset_index(drop=True)

    oldest_year = df2['date'].dt.year.min()
    most_recent_year = df2['date'].dt.year.max()
    df2['date'] = df2['date'].dt.tz_localize(None)


    repeated_rows = []

    for year in range(oldest_year, most_recent_year + 1):
        temp_df = final_result_df.copy()
        
        if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            temp_df['date'] = temp_df['date'].apply(lambda x: pd.Timestamp(f"{year}-{x}"))

        else:
            temp_df = temp_df[temp_df['date'] != '02-29']
            temp_df['date'] = temp_df['date'].apply(lambda x: pd.Timestamp(f"{year}-{x}"))

        repeated_rows.append(temp_df)

    new_df = pd.concat(repeated_rows).reset_index(drop=True)

    df_with_temps = df2.merge(new_df, on='date', how='left')

    df_with_temps['date'] = pd.to_datetime(df_with_temps['date'])

    df_with_temps['year'] = df_with_temps['date'].dt.year
    df_with_temps['month'] = df_with_temps['date'].dt.month

    avg_temp = df_with_temps.groupby('year')['temperature_2m_mean'].mean().reset_index()
    avg_temp.columns = ['year', 'avg_temp']

    spring_months = [3, 4, 5, 6]
    avg_spring_temp = df_with_temps[df_with_temps['month'].isin(spring_months)].groupby('year')['temperature_2m_mean'].mean().reset_index()
    avg_spring_temp.columns = ['year', 'avg_spring_temp']

    winter_df = df_with_temps[(df_with_temps['month'].isin([10, 11, 12]))].copy()
    winter_df['year'] = winter_df['year'] + 1

    winter_df = pd.concat([winter_df, df_with_temps[df_with_temps['month'].isin([1, 2])]])

    avg_winter_temp = winter_df.groupby('year')['temperature_2m_mean'].mean().reset_index()
    avg_winter_temp.columns = ['year', 'avg_winter_temp']

    df_with_temps = df_with_temps.merge(avg_temp, on='year', how='left')
    df_with_temps = df_with_temps.merge(avg_spring_temp, on='year', how='left')
    df_with_temps = df_with_temps.merge(avg_winter_temp, on='year', how='left')

    results = []

    for year in range(df_with_temps['year'].min(), df_with_temps['year'].max() + 1):
        df_year = df_with_temps[(df_with_temps['year'] == year) | (df_with_temps['year'] == year - 1)]
        
        sept_to_june = (df_year['month'] >= 9) | (df_year['month'] <= 6)
        march_to_june = (df_year['month'] >= 3) & (df_year['month'] <= 6)
        oct_to_feb = (df_year['month'] >= 10) | (df_year['month'] <= 2)
        
        hwn, hwm = calculate_wave_metrics(df_year, sept_to_june, 'temperature_2m_max', 'Tmax', 'heat')
        cwn, cwm = calculate_wave_metrics(df_year, sept_to_june, 'temperature_2m_min', 'Tmin', 'cold')
        shwn, shwm = calculate_wave_metrics(df_year, march_to_june, 'temperature_2m_max', 'Tmax', 'heat')
        wcwn, wcwm = calculate_wave_metrics(df_year, oct_to_feb, 'temperature_2m_min', 'Tmin', 'cold')
        
        results.append({
            'year': year,
            'HWN': hwn,
            'HWM': hwm,
            'CWN': cwn,
            'CWM': cwm,
            'SHWN': shwn,
            'SHWM': shwm,
            'WCWN': wcwn,
            'WCWM': wcwm
        })
    
    wave_metrics_df = pd.DataFrame(results)
    final_df = df_with_temps.merge(wave_metrics_df, on='year', how='left')
    final_df[['year', 'HWN', 'HWM', 'CWN', 'CWM', 'SHWN', 'SHWM', 'WCWN', 'WCWM']].drop_duplicates()

    final_df = final_df.drop(columns=['month', 'temperatures_max', 'temperatures_min'])
    cols = list(final_df.columns)
    cols.insert(1, cols.pop(cols.index('year')))
    final_df = final_df[cols]

    df_with_temps_cleaned = final_df.drop(columns=['date', 'temperature_2m_mean', 'temperature_2m_max', 'temperature_2m_min'])

    df_yearly_summary = df_with_temps_cleaned.groupby('year').mean().reset_index()

    columns_to_round = ['avg_temp', 'avg_winter_temp', 'Tmax', 'Tmin', 'avg_spring_temp', 'HWM', 'CWM', 'SHWM', 'WCWM']
    df_yearly_summary[columns_to_round] = df_yearly_summary[columns_to_round].round(2)

    return df_yearly_summary



def get_country_coordinates(country):
    geolocator = Nominatim(user_agent="GoogleV3")
    country_coordinates = {}

    try:
        location = geolocator.geocode(country)
        if location:
            country_coordinates[country] = (location.latitude, location.longitude)
        else:
            country_coordinates[country] = None
    except Exception as e:
        country_coordinates[country] = None
        print(f"Error finding coordinates for {country}: {e}")

    return country_coordinates

def read_exceptions(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def get_temperatures(country, start, end,  window_size = 7, exceptions_file='exceptions.txt') :

    exceptions = read_exceptions(exceptions_file)

    lat, long = None, None
    if exceptions:
        for exception in exceptions:
            if country in exception:
                lat = exception[country]['lat']
                long = exception[country]['long']
                print(f"Using exception coordinates for {country}: Latitude = {lat}, Longitude = {long}")
                break  


    if lat is None or long is None:
        coordinates = get_country_coordinates(country)
        for _, coord in coordinates.items():
            if coord:
                lat = coord[0]
                long = coord[1]
                print(f"Coordinates for {country} from geocoder: Latitude = {lat}, Longitude = {long}")
            else:
                print(f"Coordinates for {country} not found.")
                return None  

    df, df2 = get_data(lat, long, start = start, end = end)
    df_yearly_summary = process_data(df, df2, start = start, end =end, window_size=window_size)

    df_yearly_summary['country'] = country
    df_yearly_summary = df_yearly_summary[['country'] + [col for col in df_yearly_summary.columns if col != 'country']]

    return df_yearly_summary
