
import requests
import json
import csv
date=20230401
site_html= requests.get('https://api.weather.com/v1/location/VNKT:9:NP/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate='+str(date)+'&endDate='+str(date))#this doesn't give the html but gives the request object
data = site_html.text  # Paste the entire JSON data here

# Convert the JSON data into a Python dictionaryimport csv
 # Paste the entire JSON data here

# Convert the JSON data into a Python dictionary
response = json.loads(data)

# Extract the observations from the response
observations = response["observations"]

# Name for the output CSV file
output_csv_file = "output.csv"

# Discard keys and values with null values
cleaned_observations = []
for observation in observations:
    cleaned_observation = {key: value for key, value in observation.items() if value is not None}
    cleaned_observations.append(cleaned_observation)

# Write the cleaned observations to the CSV file
with open(output_csv_file, "a", newline="") as csvfile:
    csv_writer = csv.DictWriter(csvfile, fieldnames=cleaned_observations[0].keys())
    # csv_writer.writeheader()
    csv_writer.writerows(cleaned_observations)

print(f"CSV file '{output_csv_file}' has been created.")