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

def fitGraph(temp1, temp2):
    matrixA = np.zeros((len(temp1),2))
    matrixB = temp1

    for i in range(len(temp1)):
        matrixA[i][0] = 1
        matrixA[i][1] = temp2[i]
        
    betas = np.linalg.lstsq(matrixA,matrixB, rcond=None)
    alpha = betas[0][0]
    beta = betas[0][1]
    return alpha, beta

def drawCorrelationGraph(temp1, temp2, alpha, beta):
    t0 = temp1
    t1 = temp2
    
    plt.figure(figsize=(5,5))
    plt.scatter(t1,t0)
    plt.plot(t1, t1*beta+alpha)
    plt.ylabel("temp1")
    plt.xlabel("temp2")
    
    plt.grid()
    plt.show()
    
def frequencyOfMatchingGraph(sigma, betas):
    s = np.zeros(0)
    count = 0
    for i in range(1000):
        t = np.zeros(len(temp))
        for i in range(5):
            t[i] = temp[i]
    
        e = np.random.normal(0, sigma, 170)
        for i in range(5,len(temp)):
            t[i] = betas[0] * t[i-1]+ betas[1] * t[i-4] + e[i]
            
        alpha, beta = fitGraph(temp, t)
        s = np.append(s, beta)
        if beta > 0.5:
            count += 1
    print(count/1000)
    return s
        

def getAB(getAllFive):
    tempA = np.zeros(0)
    if getAllFive:
        tempA = np.zeros([len(year)-5,6])
        for i in range(len(year)-5):
            tempA[i][0] = 1
            tempA[i][1] = temp[i+4]
            tempA[i][2] = temp[i+3]
            tempA[i][3] = temp[i+2]
            tempA[i][4] = temp[i+1]
            tempA[i][5] = temp[i]
    
    else:
        tempA = np.zeros([len(year)-5,3])
        for i in range(len(year)-5):
          tempA[i][0] = 1
          tempA[i][1] = temp[i+4]
          tempA[i][2] = temp[i+1]
          
    tempB = np.zeros(len(year)-5)
    for i in range(len(year)-5):
      tempB[i] = temp[i+5]
    
    linearFitting = np.linalg.lstsq(tempA,tempB, rcond=None)
    alpha, betas = linearFitting[0][0], linearFitting[0][1:]
    return (alpha, betas)
    
#uncomment to see the t1 to t5
print(getAB(False))

alpha, betas = getAB(False)
t0 = temp[0:166]
t1 = temp[1:167]
t2 = temp[2:168]
t3 = temp[3:169]
t4 = temp[4:170]
sigma = stat.variance(t4-alpha-betas[0]*t3-betas[1]*t0)**0.5
print(sigma)

s = frequencyOfMatchingGraph(sigma, betas)
plt.hist(s)
plt.title("Historgram of Slope after each 1000 run")
plt.xlabel('betas (slope)')
plt.ylabel('occurance out of 1000')
plt.show()

t = np.zeros(len(temp))
for i in range(5):
    t[i] = temp[i]

e = np.random.normal(0, sigma, 170)
for i in range(5,len(temp)):
    t[i] = betas[0] * t[i-1]+ betas[1] * t[i-4] + e[i]


    
f, (plot1, plot2) = plt.subplots(1, 2)
alpha, beta = fitGraph(temp, t)
print(beta)
plot1.plot(year,t, marker='.', markersize=1, color="green", label='Global Mean Temperature Deviation from 1961-1990 mean')
plot1.set_title("ar(1,4) model of global temp.series")
plot2.plot(year,temp, marker='.', markersize=1, color="green", label='Global Mean Temperature Deviation from 1961-1990 mean')
plot2.set_title("Global temperature record")
plt.tight_layout()
plt.show()
drawCorrelationGraph(temp, t, alpha, beta)

