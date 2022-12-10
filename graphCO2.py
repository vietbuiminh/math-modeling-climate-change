import numpy as np
import csv
import matplotlib.pyplot as plt

root = "data/"

csv_filename = "ghg-concentrations_fig-1.csv"
x = np.empty(0)
y = np.empty(0)

with open(root + csv_filename) as f:
    placeList = ['EPICA Dome C and  Vostok Station,  Antarctica','Law Dome, Antarctica (75-year smoothed)	Siple Station, Antarctica','Mauna Loa, Hawaii','Barrow, Alaska','Cape Matatula,  American Samoa','South Pole, Antarctica','Cape Grim, Australia','Lampedusa Island, Italy	','Shetland Islands, Scotland']
    reader = csv.DictReader(f)
    for row in reader:
        year = row.get('Year (negative values = BC)')
        sumCO2 = 0
        count = 0
        for place in placeList:
            if row.get(place) not in [None, '']:
                sumCO2 += float(row.get(place))
                count += 1
        if count != 0:
            avg = sumCO2/count
            print(year, avg)
            x = np.append(x, year)
            y = np.append(y, avg)


plt.figure(figsize=(10,5))
plt.scatter(x,y)
plt.title("Figure 1. Global Atmospheric Concentrations of Carbon Dioxide Over Time")
plt.ylabel("parts per million (ppm) of CO2")
plt.xlabel("year")
plt.show()