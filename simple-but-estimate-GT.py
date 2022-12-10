import numpy as np
import matplotlib.pyplot as plt
import statistics as stat

year = np.empty(0)
temp = np.empty(0)

fin = open('global-mean-temp.txt', 'r')
reader = fin.readline()
reader = fin.readlines()
for line in reader:
    line = line.strip()
    line = line.split(" ")
    year = np.append(year,int(float(line[0])))
    temp = np.append(temp,float(line[1]))

alpha = stat.mean(temp)
mu, sigma = 0, stat.variance(temp)**0.5
e = np.random.normal(mu,sigma, 170) #170 = 2019 - 1850 + 1
temp = alpha + e

print(sigma, alpha)

plt.figure(figsize=(10,5))
plt.plot(year,temp, marker='.', markersize=1, color="green", label='Simulation from simple GT Model')
plt.title("Simulation, but estimate, GT Model")
plt.ylabel("temp")
plt.xlabel("year")
plt.show()