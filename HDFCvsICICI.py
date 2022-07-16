import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from pytz import AmbiguousTimeError

df = pd.read_csv("hdfc_icici.csv")

data = df.iloc[:].values
data = np.transpose(data)

for i in range(741):
    if data[0][i][1] == ",":
        data[0][i] = "1" + data[0][i][2:]

data = data.astype(float)

meanX30 = []
meanY30 = []
varX30 = []
varY30 = []
stdevX30 = []
stdevY30 = []
covarXY30 = []
corXY30 = []

meanX60 = []
meanY60 = []
varX60 = []
varY60 = []
covarXY60 = []
corXY60 = []

sumX = 0
sumY = 0
sum2X = 0
sum2Y = 0
sumXY = 0

for i in range(700):
    for j in range(30):
        sumX = sumX + data[0][41+i-j]
        sumY = sumY + data[1][41+i-j]
    meanX30.append(sumX/30)
    meanY30.append(sumY/30)
    sumX = 0
    sumY = 0

for i in range(700):
    for j in range(30):
        sum2X = sum2X + pow(data[0][41+i-j],2)
        sum2Y = sum2Y + pow(data[1][41+i-j],2)
        sumXY = sumXY + ((data[0][41+i-j]-meanX30[i])*(data[1][41+i-j]-meanY30[i]))
    varX30.append((sum2X/30)-pow(meanX30[i],2))
    varY30.append((sum2Y/30)-pow(meanY30[i],2))
    covarXY30.append((sumXY/30)) 
    stdevX30.append(pow(varX30[i],0.5))
    stdevY30.append(pow(varY30[i],0.5))
    corXY30.append(covarXY30[i]/(stdevX30[i]*stdevY30[i]))
    sum2X = 0
    sum2Y = 0
    sumXY = 0

n=np.array(range(700))

n = n + 1

plt.plot(n,np.array(data[0][41:]))
#plt.plot(n,np.array(data[1][41:]))
plt.plot(n,np.array(corXY30))
plt.plot(n,np.array(meanX30)-np.array(stdevX30))
plt.plot(n,np.array(meanX30)+np.array(stdevX30))
#plt.plot(n,3*np.array(meanY30)-3*np.array(stdevY30))
#plt.plot(n,3*np.array(meanY30)+3*np.array(stdevY30))
plt.show()

def buycor(cor1,cor2):
    Amount = 0.0
    Xown = 0
    Yown = 0
    ctr1 = 0
    ctr2 = 0
    for i in range(699):
        i = i + 1
        if corXY30[i-1]>=cor1 and corXY30[i]<cor1:
            ctr1 = ctr1 + 1
            a = abs(data[0][41+i]-data[0][41+i-1])
            b = abs(data[1][41+i]-data[1][41+i-1])
            k = math.ceil(10*(data[0][41+i]/data[1][41+i]))

            if data[0][41+i]>data[0][41+i-1] and data[1][41+i]>data[1][41+i-1]:
                if a>=b:
                    Xown = Xown - 10
                    Yown = Yown + k
                    Amount = Amount + 10*data[0][41+i] - k*data[1][41+i]

                else:
                    Xown = Xown + 10
                    Yown = Yown - k
                    Amount = Amount + k*data[1][41+i] - 10*data[0][41+i]
                    
            elif data[0][41+i]<data[0][41+i-1] and data[1][41+i]<data[1][41+i-1]:
                if a>=b:
                    Xown = Xown + 10
                    Yown = Yown - k
                    Amount = Amount + k*data[1][41+i] - 10*data[0][41+i]

                else:
                    Xown = Xown - 10
                    Yown = Yown + k
                    Amount = Amount + 10*data[0][41+i] - k*data[1][41+i]

            elif data[0][41+i]>data[0][41+i-1] and data[1][41+i]<data[1][41+i-1]:
                Xown = Xown - 10
                Yown = Yown + k
                Amount = Amount + 10*data[0][41+i] - k*data[1][41+i]

            elif data[0][41+i]<data[0][41+i-1] and data[1][41+i]>data[1][41+i-1]:
                Xown = Xown + 10
                Yown = Yown - k
                Amount = Amount + k*data[1][41+i] - 10*data[0][41+i]

        if ctr1>0 and corXY30[i-1]<=cor2 and corXY30[i]>cor2:
            ctr2 = ctr2 + ctr1
            ctr1 = 0
            Amount = Amount + Xown*data[0][41+i] + Yown*data[1][41+i]
            Xown = 0
            Yown = 0

#            print(Amount)

    if ctr1>0:
        print(cor2)
        print(Xown)
        print(Yown)
        Amount = Amount + Xown*data[0][41+i] + Yown*data[1][41+i]

    return Amount,ctr1,ctr2

def buymean(k):
    Amt = 0
    Xqty = 0
    Yqty = 0
    Xbctr = 0
    Ybctr = 0
    Xsctr = 0
    Ysctr = 0

    for i in range(699):
        i = i + 1
        l = math.ceil(10*(data[0][41+i]/data[1][41+i]))

        if data[0][41+i-1] >= meanX30[i] - (k*stdevX30[i]) and data[0][41+i-1] < meanX30[i] - (k*stdevX30[i]):
            Amt = Amt - (data[0][41+i]*10)
            Xqty = Xqty + 10
            Xbctr = Xbctr + 1

        if data[1][41+i-1] >= meanY30[i] - (k*stdevY30[i]) and data[0][41+i-1] < meanY30[i] - (k*stdevY30[i]):
            Amt = Amt - (data[1][41+i]*l)
            Yqty = Yqty + l
            Ybctr = Ybctr + 1

        if data[0][41+i-1] <= meanX30[i] + (k*stdevX30[i]) and data[0][41+i-1] > meanX30[i] + (k*stdevX30[i]) and Xqty > 0:
            Amt = Amt + (data[0][41+i]*Xqty)
            Xqty = 0
            Xsctr = Xsctr + 1

        if data[1][41+i-1] <= meanY30[i] + (k*stdevY30[i]) and data[0][41+i-1] > meanY30[i] + (k*stdevY30[i]) and Yqty > 0:
            Amt = Amt + (data[1][41+i]*Yqty)
            Yqty = 0
            Ysctr = Ysctr + 1

    if Xqty > 0:
        print(k)
        print(Xqty)
        Amt = Amt + (data[0][41+i]*Xqty)

    if Yqty > 0:
        print(k)
        print(Yqty)
        Amt = Amt + (data[0][41+i]*Yqty)

    return Amt
    
#print(buy(0.8))

a = np.linspace(0.3,0.8,51)

c = []
for ele in a : c.append(buycor(0.3,ele))

plt.plot(a,c)

plt.show()

print(buymean(0.1))

#print(buy(0.4,0.6))

#print(corXY30)