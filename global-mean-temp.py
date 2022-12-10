import numpy as np
import csv
import matplotlib.pyplot as plt
import statistics as stat

root = "data/"

csv_filename = "temperature-anomaly.csv"
wFile = open("global-mean-temp.txt", 'w')
wFile.write("Year Temp\n")


x = np.empty(0)
y = np.empty(0)

with open(root + csv_filename) as f:
  reader = csv.DictReader(f)
  for row in reader:
    if row.get('Entity') == "Global":
      year = float(row.get('Year'))
      temp = float(row.get('Median temperature anomaly from 1961-1990 average'))
      x = np.append(x,year)
      y = np.append(y,temp)
      wFile.write(f'{year} {temp}\n')
wFile.close()

plt.figure(figsize=(10,5))
plt.plot(x,y, marker='.', markersize=1, color="lightblue", label='Global Mean Temperature Deviation from 1961-1990 mean')
plt.title("Global mean temperarutre deviations from 1961-1990 mean")
plt.ylabel("temp")
plt.xlabel("year")
plt.show()
print(x[0],x[len(x)-1])


