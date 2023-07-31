import requests
import json
import csv
import datetime

# Name for the output CSV file
output_csv_file = "output.csv"

# Create or clear the CSV file and write the header row
with open(output_csv_file, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow([])  # This will create an empty row as a placeholder for the header

# Loop through each date from 2020 to 2022
start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2022, 12, 31)

current_date = start_date
while current_date <= end_date:
    try:
        # Format the date as YYYYMMDD
        date = current_date.strftime("%Y%m%d")

        # Make the API request
        site_html = requests.get(f'https://api.weather.com/v1/location/VNKT:9:NP/observations/historical.json?apiKey=e1f10a1e78da46f5b10a1e78da96f525&units=e&startDate={date}&endDate={date}')
        data = site_html.text

        # Convert the JSON data into a Python dictionary
        response = json.loads(data)

        # Check if there are observations for the current date
        if "observations" in response:
            observations = response["observations"]

            # Append the observations to the CSV file
            with open(output_csv_file, "a", newline="") as csvfile:
                csv_writer = csv.DictWriter(csvfile, fieldnames=observations[0].keys())
                if current_date == start_date:
                    csv_writer.writeheader()
                csv_writer.writerows(observations)

            print(f"Data for {date} has been added to '{output_csv_file}'.")
        else:
            print(f"No data available for {date}.")
        
    except Exception as e:
        print(f"Error fetching data for {date}: {str(e)}")

    # Move to the next date
    current_date += datetime.timedelta(days=1)

print("Data retrieval completed.")
