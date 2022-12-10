import numpy as np
import matplotlib.pyplot as plt
import csv
import statistics as stat

root = "data/"

csv_filename = "temperature-anomaly.csv"
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


diff = np.empty(0)
for i in range(y.size-1):
  diff = np.append(diff, y[i+1] - y[i])
  
sigma = stat.variance(diff)**0.5 

temp = np.empty(0)
temp = np.append(temp,y[0])  

e = np.random.normal(0,sigma, 170) 

for i in range(1, e.size):
  temp = np.append(temp, temp[i-1]+e[i-1])
  
plt.figure(figsize=(10,5))
plt.plot(x,temp, marker='.', markersize=1, color="green", label='Random Walk GT Model')
plt.title("Random Walk GT model")
plt.ylabel("temp")
plt.xlabel("year")
plt.show()
