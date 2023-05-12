import tensorflow as tf
import requests
import numpy as np

model = tf.keras.models.load_model('Model/model_temp.h5')

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

input = np.array([[humidity, wind_speed, wind_dir, precip, pressure, app_temp, cloud, hour, temp]], dtype=np.float32)

print(input)

input = input.reshape(1,9,1)

# check nan value in input and replace with mean value
for i in range(9):
    if np.isnan(input[0][i][0]):
        input[0][i][0] = 0

y_pred = model.predict(input)

print(y_pred)