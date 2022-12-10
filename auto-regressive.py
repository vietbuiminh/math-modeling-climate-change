import numpy as np
import csv
import matplotlib.pyplot as plt
import statistics as stat

x = np.empty(0)
y = np.empty(0)

fin = open('global-mean-temp.txt', 'r')
reader = fin.readline()
reader = fin.readlines()
for line in reader:
    line = line.strip()
    line = line.split(" ")
    x = np.append(x,int(float(line[0])))
    y = np.append(y,float(line[1]))


def tempThisVSLastYear():
    t0 = y[1:170]
    t1 = y[0:169]
    
    beta, alpha = np.polyfit(t0,t1,1)
    sigma_0 = stat.variance(t0-alpha-beta*t1)
    plt.figure(figsize=(6,5))
    plt.scatter(t1,t0)
    plt.plot(t1, t1*beta+alpha)
    plt.ylabel("temp this year")
    plt.xlabel("temp last year")
    plt.grid()
    plt.show()
    return sigma_0, alpha, beta
  
def ar1Model():
  sigma, a, b = tempThisVSLastYear()
  print(sigma,a,b)
  temp = np.empty(0)
  temp = np.append(temp,y[0])  # set first element to be observe the first year of the y data
  e = np.random.normal(0,sigma**0.5, 170) #simulate random temperature changes
  for i in range(1, e.size):
    temp = np.append(temp, a + b*temp[i-1]+e[i-1])
  plt.figure(figsize=(10,5))
  plt.plot(x,temp, marker='.', markersize=1, color="purple", label='Random Walk GT Model')
  plt.title("ar(1) Model")
  plt.ylabel("temp")
  plt.xlabel("year")
  plt.show()



ar1Model()