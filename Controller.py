import requests
import tensorflow as tf
import numpy as np
import warnings

warnings.filterwarnings("ignore")


url = "https://api.weatherbit.io/v2.0/current?&city=Hanoi&country=VN&key=ccc9d9071252498cbd589c8689961bd8&include=minutely"

response = requests.get(url)
data = response.json()

# get data humidity,wind_speed,wind_dir,precip,pressure,app_temp,cloud,hour,month,temp

humidity = data['data'][0]['rh']
wind_speed = data['data'][0]['wind_spd']
wind_dir = data['data'][0]['wind_dir']
precip = data['data'][0]['precip']
pressure = data['data'][0]['pres']
app_temp = data['data'][0]['app_temp'] + 273.15
cloud = data['data'][0]['clouds']
hour = int(data['data'][0]['ob_time'][8:10])
temp = data['data'][0]['temp'] + 273.15
weather = data['data'][0]['weather']['description']

x_mean = np.array([7.74973094e+01, 2.80202640e+00, 1.42257474e+02, 2.46415703e-01,
       1.00962442e+03, 3.00062242e+02, 6.81242020e+01, 1.14995508e+01,
       2.97592885e+02])
x_std = np.array([ 15.88368275,   1.40884157, 101.64481521,   0.99006853,
         7.03036941,   8.3542265 ,  27.69297933,   6.92203511,
         5.75967504])

input = np.array([[humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp]], dtype=np.float32)

for i in range(9):
    # check if input nan 
    if np.isnan(input[0][i]):
        input[0][i] = x_mean[i]

def predict_temp():
    model = tf.keras.models.load_model('Model/model_temp.h5')
   
    X = (input- x_mean)/x_std
    y_pred = model.predict(X)
    y_pred = y_pred*5.759 + 24.44
    y_pred = np.round(y_pred,1, out=None)
    print(y_pred)
    return y_pred[0]

def get_icon_url(weather):
    if weather == 1:
        return 'url(:/Img/Image/icons8-rain-48.png);'
    elif weather == 2:
        return 'url(:/Img/Image/icons8-partly-cloudy-day-48.png);'
    elif weather == 3:
        return 'url(:/Img/Image/icons8-sun-48.png);'
    elif weather == 0:
        return 'url(:/Img/Image/fog.png);'
    
def predict_weather_1h():
    model = tf.keras.models.load_model('Model/predict_weather.h5')
    # X = np.array([[humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp]], dtype=np.float32)
    y_pred = model.predict(input)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_2h():
    model = tf.keras.models.load_model('Model/predict_weather2.h5')
    # X = np.array([[humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp]], dtype=np.float32)
    y_pred = model.predict(input)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_3h():
    model = tf.keras.models.load_model('Model/predict_weather3.h5')
    # X = np.array([[humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp]], dtype=np.float32)
    y_pred = model.predict(input)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_4h():
    model = tf.keras.models.load_model('Model/predict_weather4.h5')
    # X = np.array([[humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp]], dtype=np.float32)
    y_pred = model.predict(input)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)

def predict_weather_5h():
    model = tf.keras.models.load_model('Model/predict_weather5.h5')
    # X = np.array([[humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp]], dtype=np.float32)
    y_pred = model.predict(input)
    y_pred = np.argmax(y_pred)
    return get_icon_url(y_pred)


def get_icon_now():
    if weather == 'Overcast clouds' or weather == 'Broken clouds' or weather == 'Broken clouds' :
        return 'url(:/Img/Image/icons8-partly-cloudy-day-48.png);'
    
    elif weather == 'Clear sky' or weather == 'Few clouds' or weather == 'Scattered clouds' :
        return 'url(:/Img/Image/icons8-sun-48.png);'
    
    elif weather == 'Fog' or weather == 'Haze':
        return 'url(:/Img/Image/fog.png);'
    else:
        return 'url(:/Img/Image/icons8-rain-48.png);'
    