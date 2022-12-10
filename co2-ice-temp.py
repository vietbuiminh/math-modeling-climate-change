import numpy as np
import matplotlib.pyplot as plt

root = "data/"

iceDDTemp = np.empty(0)
iceAge = np.empty(0)
iceFile = open(root+"vostok-dd.txt")
iceFileList = iceFile.readlines()
for i in range(5, len(iceFileList)):
    line = iceFileList[i][:-2]
    line = line.split()
    age = float(line[1]) // 1000
    temp = float(line[-1]) 
    iceDDTemp = np.append(iceDDTemp, temp)
    iceAge = np.append(iceAge, age)
iceFile.close()

co2 = np.empty(0)
iceAge2 = np.empty(0)
co2File = open(root+"vostok-co2.txt")
co2FileList = co2File.readlines()
for i in range(5, len(co2FileList)):
    # print(co2FileList[i])
    line = co2FileList[i].strip()[:-1]
    line = line.split()
    age = float(line[1]) // 1000
    co2Level = float(line[-1]) 
    co2 = np.append(co2, co2Level)
    iceAge2 = np.append(iceAge2, age)
# print(len(iceAge), len(iceAge2))
    
f, (plot1, plot2) = plt.subplots(2, 1)

plot1.plot(iceAge,iceDDTemp, marker='.', markersize=1, color="green", label='Graph1')
plot1.set_title("Vostok Ice Core D Based Temperature")
plot1.set_ylabel("t")
plot1.set_xlabel("age in 1000 years")
# plot1.xlabel("year")

plot2.plot(iceAge2,co2, marker='.', markersize=1, color="green", label='Graph2')
plot2.set_title("Vostok Ice Core CO2 Record")
plot2.set_ylabel("CO2")
plot2.set_xlabel("age in 1000 years")
# plot2.xlabel("year")

plt.tight_layout()
temp0 = np.zeros(len(iceAge2))
for i in range(len(iceAge2)):
    # print(np.where(iceAge == iceAge2[i])[0][0])
    temp0[i] = iceDDTemp[np.where(iceAge == iceAge2[i])[0][0]]
    
def getAlphaBeta(temp1, temp2):
    matrixA = np.zeros((len(temp1),2))
    matrixB = temp1
    
    for i in range(len(temp1)):
        matrixA[i][0] = 1
        matrixA[i][1] = temp2[i]
        
    betas = np.linalg.lstsq(matrixA,matrixB, rcond=None)
    alpha = betas[0][0]
    beta = betas[0][1]
    return alpha, beta
alpha, beta = getAlphaBeta(temp0, co2)
print(alpha, beta)

def runHistogram():
    s = np.zeros(1000)
    for i in range(1000):
        shuffledco2 = co2
        np.random.shuffle(shuffledco2)
        a,b = getAlphaBeta(temp0, shuffledco2)
        s[i] = b
    plt.figure(0)
    plt.hist(s)
    plt.title("Distribution of beta is we shuffle CO2")
    plt.xlabel('betas (slope)')
    plt.ylabel('occurrance out of 1000')
    plt.show()
    
def temp0vsiceDDTemp():
    t0 = temp0
    t1 = co2
    plt.figure(0)
    plt.figure(figsize=(6,5))
    plt.scatter(t1,t0)
    plt.plot(t1, t1*beta+alpha)
    plt.ylabel("temp")
    plt.xlabel("co2")
    plt.title("Vostok Temp vs CO2 overthe last 400K+ years")
    plt.grid()
    plt.show()
    
temp0vsiceDDTemp()
runHistogram()
#shuffled data
fig, axs = plt.subplots(2, 3, constrained_layout=True)
def addPlot(ax,x,y,a,b):
    ax.scatter(y,x, s=10, c='g')
    ax.plot(y, y*b+a)
    ax.set_title(f'beta = {round(b,4)}')
    ax.set_ylabel("y")
    ax.set_xlabel("x")
addPlot(axs[0][0],temp0,co2,alpha,beta)
for i in range(2):
    shuffledco2 = co2
    np.random.shuffle(shuffledco2)
    a,b = getAlphaBeta(temp0, shuffledco2)
    addPlot(axs[0][i+1],temp0,shuffledco2,a,b)
for i in range(3):
    shuffledco2 = co2
    np.random.shuffle(shuffledco2)
    a,b = getAlphaBeta(temp0, shuffledco2)
    addPlot(axs[1][i],temp0,shuffledco2,a,b)
# run 1000 to see the distribution of beta

