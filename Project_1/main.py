import numpy as np
import scipy.io

data = scipy.io.loadmat('mnist_data.mat')
i = 0
'''
Transferring all the data into numpy array and taking its mean and standard deviation
'''
tr_x = data['trX']
tr_y = data['trY']
ts_x = data['tsX']
ts_y = data['tsY']
ts_yT = ts_y.T
tr_xm = np.mean(tr_x, axis=1)
tr_xsd = np.std(tr_x, axis=1)
ts_xm = np.mean(ts_x, axis=1)
ts_xsd = np.std(ts_x, axis=1)
nof_ones = np.ones(tr_xm.size)
nof_ones_ts = np.ones(ts_yT.size)
#print(nof_ones.shape)
#print(nof_ones_ts.shape)
#print(tr_xm.shape)
#print(ts_yT.shape)
tr_x_new = np.vstack((tr_xm,tr_xsd,nof_ones)).T
#print(tr_x_new.shape)
ts_x_new = np.vstack((ts_xm,ts_xsd,nof_ones_ts)).T
#print(ts_x_new.shape)

theta = np.zeros([1,3])         #Defining array of zeros
k = theta.T
#print(theta.shape)
#print(k.shape)

learning_rate = 0.001           # Initializing Learning Rate
#defining sigmoid function
def sigmoid(c):
    return 1.0/(1.0 + np.exp(-c))

#running the loop with the number of epochs i and storing the result in theta

while(i<100000):
    theta = theta + learning_rate * np.dot(tr_y, tr_x_new) - learning_rate * np.dot(sigmoid((np.dot(theta, np.transpose(tr_x_new)))), tr_x_new)
    i = i + 1

output = 0
output = np.transpose(sigmoid(np.dot(theta, np.transpose(ts_x_new))))
cnt = 0

'''
Now dividing the output and sorting them in two sets of digit 7 & 8
'''
for i in range(ts_y.size):
    if(output[i]>=0.5):
        output[i] = 1
    else:
        output[i] = 0
cnt7=0
cnt8=0

# Calculate accuracy
for i in range(ts_y.size):
    if(ts_yT[i] == output[i]):
        cnt += 1

for i in range(0,1028):
    if (ts_yT[i] == output[i]):
        cnt7 += 1

for i in range(1028,2002):
    if (ts_yT[i] == output[i]):
        cnt8 += 1

print(cnt/2002)                         # overall accuracy
print(cnt7/1028)                        # Digit 7 accuracy
print(cnt8/974)                         # Digit 8 accuracy