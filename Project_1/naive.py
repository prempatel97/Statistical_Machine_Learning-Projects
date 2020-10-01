import numpy as np
import pandas as pd
import math
import scipy.io
data = scipy.io.loadmat('mnist_data.mat')

'''
Transferring all the data into numpy array and taking its mean and standard deviation
'''

tr_x = np.array(data['trX'])
ts_x = np.array(data['tsX'])
tr_y = np.array(data['trY'])
ts_y = np.array(data['tsY'])
tr_xm = tr_x.mean(axis = 1)
tr_xsd = tr_x.std(axis = 1)
#print(tr_x.shape)
#print(ts_x.shape)
#print(tr_y.shape)
#print(ts_y.shape)
#print(tr_xm.shape)
#print(tr_xsd.shape)

#Dividing two data into two sets of digit 7 & digit 8

trxm_d7 = tr_xm[:6265]
#print(trxm_d7.shape)
trxsd_d7 = tr_xsd[:6265]
#print(trxsd_d7.shape)
trxm_d8 = tr_xm[6265:]
#print(trxm_d8.shape)
trxsd_d8 = tr_xsd[6265:]
#print(trxsd_d8.shape)

#Defining the formula for calculating gaussian probability

def NaiveBayesProb(x,mean,std):
    t=((x-mean)**2)/(2*std**2)
    exp=(np.e)**- t
    return ((exp/(std*(2*np.pi)**0.5)))

# Calculating the probability of digit 7
def probofset7(v):
    answer = 1
    for cnt, i in enumerate(v):
        answer *= NaiveBayesProb(i,trxm_d7[cnt],trxsd_d7[cnt])
    return answer

# Calculating the probability of digit 7

def probofset8(v):
    answer = 1
    for cnt, i in enumerate(v):
        answer *= NaiveBayesProb(i,trxm_d8[cnt],trxsd_d8[cnt])
    return answer

# Now predicting the results whether the data which is fed is of digit 7 and digit 8

def predict_data(ts_X):
    answer = []
    for i in ts_X:
        prob_set_7 = probofset7(i)
        prob_set_8 = probofset8(i)
        if prob_set_7 > prob_set_8:
            answer.append(0.0)
        else:
            answer.append(1.0)
    return answer

finalAns = predict_data(ts_x)

# Finding the overall accuracy of the testing data

n = np.sum(finalAns == ts_y)
accr = n/ts_y.size
print(accr)             #Printing the overall accuracy

cnt7=0
cnt8=0

# Calculating accuracy of individual set of digit 7

for i in range(1028):
    if(finalAns[i] == ts_y[0,i]):
        cnt7 += 1

# Calculating accuracy of individual set of digit 7

for i in range(1028,2002):
    if(finalAns[i] == ts_y[0,i]):
        cnt8 += 1

print(cnt7/1028)        #Printing the digit 7 accuracy
print(cnt8/974)         #Printing the digit 8 accuracy