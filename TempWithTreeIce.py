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
      x = np.append(x,int(year))
      y = np.append(y,temp)

def getXY(file, type_split, columnNum):
  x = np.empty(0)
  y = np.empty(0)
  for line in file.readlines():
    data = line.split(type_split)
    #print(data)
    x = np.append(x, int(float(data[0])))
    y = np.append(y, float(data[columnNum]))
  return x,y

# Ice DD
openFile = open(root+"james-ross-island2013dd.txt", 'r')
titles = openFile.readline()
years1, ALL_50_full = getXY(openFile, "\t", 2)
openFile.close()
# Tree Width
openFile = open(root+"treewidth_frenchalps.txt", 'r')
titles = openFile.readline()
years2, width_index = getXY(openFile, "\t", 1)
openFile.close()

print(int(years1[0]), int(years1[-1]))
print(int(years2[0]), int(years2[-1]))

years = np.arange(1900, 2001, 1)
TRW = np.zeros(len(years))
IDD = np.zeros(len(years))
TEMP = np.zeros(len(years))

def tempThisVSLastYear():
  t0 = TRW
  t1 = IDD
  
  beta, alpha = np.polyfit(t0,t1,1)
  sigma_0 = stat.variance(t0-alpha-beta*t1)
  print(sigma_0)
  plt.figure(figsize=(6,5))
  plt.scatter(t1,t0)
  plt.ylabel("ice dd")
  plt.xlabel("tree width index")
  a, b = alpha, beta
  plt.grid()
  plt.show()
  print(a,b)

for i in range(107, 6, -1):
    IDD[107-i] = ALL_50_full[i]
for i in range(len(years)):
    TRW[i] = width_index[i+931]
    TEMP[i] = y[i+50]

tempThisVSLastYear()
matrixA = np.zeros((len(years),3))
matrixB = np.zeros(len(years))

for i in range(len(years)):
    matrixA[i][0] = 1
    matrixA[i][1] = TRW[i]
    matrixA[i][2] = IDD[i]
    matrixB[i] = TEMP[i]
    
betas = np.linalg.lstsq(matrixA,matrixB, rcond=None)
alpha = betas[0][0]
beta = betas[0][1]
lambd = betas[0][2]
print(alpha, beta, lambd)
# Finding temperature from 1400 to 1900
e = np.random.normal(0, alpha**0.5, 501)
for i in range(500):
    prevTemp = beta*width_index[i+930-i*2] + lambd*ALL_50_full[i+108] + alpha
    # print(years2[i+930-i*2],years1[i+108])
    # print(years1[i+108])
    TEMP = np.insert(TEMP, 0, prevTemp)
    years = np.insert(years, 0, years1[i+108])


plt.figure(figsize=(10,5))
plt.plot(years,TEMP, marker='.', markersize=1, color="lightblue", label='Global Mean Temperature Deviation from 1961-1990 mean')
plt.title("Northern Hemisphere reconstructed temperature anomalies")
plt.ylabel("temp")
plt.xlabel("year")
plt.show()
  