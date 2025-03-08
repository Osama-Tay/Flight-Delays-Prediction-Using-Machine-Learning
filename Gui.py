import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import joblib
import numpy as np

model = joblib.load('decision_tree.pkl')
weather_condition_encoder=joblib.load('WeatherCondition_encoder.pkl')
carrier_encoder=joblib.load('Carrier_encoder.pkl')
wind_direction_encoder=joblib.load('WindDirection_encoder.pkl')
destination_encoder=joblib.load('Destination_encoder.pkl')

def calculate_minutes(time) -> int:
    minutes=time%100
    hours=(time//100) * 60
    return minutes+hours

def predict(data):
    return model.predict(data)

def show_data():
    # Get the selected month
    selected_month = month_var.get()
    selected_day_of_month = int(day_of_month_entry.get())
    selected_day_of_week = int(day_of_week_entry.get())
    departure_time=calculate_minutes(int(dep_time.get()))
    arrival_time_data=calculate_minutes(int(arr_time.get()))
    weather_condition_data = weather_condition_encoder.transform([str(weather_var.get())])[0]
    carrier_data = carrier_encoder.transform([str(carrier.get())])[0]
    wind_direction_data = wind_direction_encoder.transform([str(wind_direction.get())])[0]
    destination_data = destination_encoder.transform([str(destination.get())])[0]
    journey_time_data=int(journey_time.get())
    wind_speed_data=int(wind_speed.get())
    distance_data=int(distance.get())
    pressure_data=int(pressure.get())
    dew_point_data=int(dew_point.get())
    wind_gust_data=int(wind_gust.get())
    humidity_data=int(humidity.get())
    temprature_data=int(temprature.get())
    no_arrivals=int(no_arrive.get())
    no_deps=int(no_departures.get())

    data=np.array([departure_time, arrival_time_data,
       journey_time_data, distance_data, destination_data, carrier_data,
       no_arrivals, no_deps, pressure_data,
       temprature_data, dew_point_data, wind_speed_data, humidity_data, wind_direction_data,
       selected_day_of_month, weather_condition_data, selected_day_of_week, selected_month, wind_gust_data,1,1]).reshape(1,-1)

    predictions=predict(data)

    if predictions==0:
        result_label.config(text="Your flight would probably depart on time")
    else:
        result_label.config(text="Your flight would probably be delayed")

if __name__=='__main__':
    # Create the main window
    root = tk.Tk()
    root.title("Flight Data GUI")
    root.geometry("1000x1000")  # Set the size of the window

    # Label for the month
    month_label = ttk.Label(root, text="Month:")
    month_label.pack()

    # Radio buttons for selecting the month
    month_var = tk.IntVar()
    month_var.set(11)  # Default to November
    month_frame = ttk.Frame(root)
    month_frame.pack()
    for month in [11, 12, 1]:
        ttk.Radiobutton(month_frame, text=str(month), variable=month_var, value=month).pack(side="left")

    weather_label = ttk.Label(root, text="Weather Condition:")
    weather_label.pack()
    weather_var= tk.StringVar()
    weather_var.set('Fair')  # Default to November
    weather_frame = ttk.Frame(root)
    weather_frame.pack()
    for month in ['Fair / Windy', 'Fair', 'Light Rain / Windy', 'Partly Cloudy',
       'Mostly Cloudy', 'Cloudy', 'Light Rain', 'Mostly Cloudy / Windy',
       'Partly Cloudy / Windy', 'Light Snow / Windy', 'Cloudy / Windy',
       'Light Drizzle', 'Rain', 'Heavy Rain', 'Fog', 'Wintry Mix',
       'Light Freezing Rain', 'Light Snow', 'Wintry Mix / Windy',
       'Fog / Windy', 'Light Drizzle / Windy', 'Rain / Windy',
       'Drizzle and Fog', 'Snow', 'Heavy Rain / Windy']:
        ttk.Radiobutton(weather_frame, text=str(month), variable=weather_var, value=month).pack(side="left")

    # Inputs for day of the month and day of the week
    day_of_month_label = ttk.Label(root, text="Day of Month:")
    day_of_month_label.pack()
    day_of_month_entry = ttk.Entry(root)
    day_of_month_entry.pack()

    day_of_week_label = ttk.Label(root, text="Day of Week:")
    day_of_week_label.pack()
    day_of_week_entry = ttk.Entry(root)
    day_of_week_entry.pack()

    no_departures = ttk.Label(root, text="Number of departing flights:")
    no_departures.pack()
    no_departures = ttk.Entry(root)
    no_departures.pack()

    dep_time = ttk.Label(root, text="Departure Time:")
    dep_time.pack()
    dep_time = ttk.Entry(root)
    dep_time.pack()

    arr_time = ttk.Label(root, text="Arrival Time:")
    arr_time.pack()
    arr_time = ttk.Entry(root)
    arr_time.pack()

    journey_time = ttk.Label(root, text="Expected Journey Time (in minutes):")
    journey_time.pack()
    journey_time = ttk.Entry(root)
    journey_time.pack()

    no_arrive = ttk.Label(root, text="Number of arrival flights:")
    no_arrive.pack()
    no_arrive = ttk.Entry(root)
    no_arrive.pack()

    distance = ttk.Label(root, text="Distance:")
    distance.pack()
    distance = ttk.Entry(root)
    distance.pack()

    destination = ttk.Label(root, text="Destination:")
    destination.pack()
    destination = ttk.Entry(root)
    destination.pack()

    carrier = ttk.Label(root, text="Carrier:")
    carrier.pack()
    carrier = ttk.Entry(root)
    carrier.pack()

    pressure = ttk.Label(root, text="Air Pressure:")
    pressure.pack()
    pressure = ttk.Entry(root)
    pressure.pack()

    temprature = ttk.Label(root, text="Temprature:")
    temprature.pack()
    temprature = ttk.Entry(root)
    temprature.pack()

    wind_speed = ttk.Label(root, text=" Wind Speed:")
    wind_speed.pack()
    wind_speed = ttk.Entry(root)
    wind_speed.pack()

    dew_point = ttk.Label(root, text="Dew Point:")
    dew_point.pack()
    dew_point = ttk.Entry(root)
    dew_point.pack()

    humidity = ttk.Label(root, text="Humidity:")
    humidity.pack()
    humidity = ttk.Entry(root)
    humidity.pack()

    wind_direction = ttk.Label(root, text="Wind Direction:")
    wind_direction.pack()
    wind_direction = ttk.Entry(root)
    wind_direction.pack()

    wind_gust = ttk.Label(root, text="Wind Gusts:")
    wind_gust.pack()
    wind_gust = ttk.Entry(root)
    wind_gust.pack()
    
    # Button to show data
    show_data_button = ttk.Button(root, text="Show Data", command=show_data)
    show_data_button.pack()

    # Label to display the result
    result_label = ttk.Label(root, text="")
    result_label.pack()

    root.mainloop()

