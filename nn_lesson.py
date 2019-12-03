import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


data_inputs = np.array([[3, 5], [5, 1], [10, 2]])
data_outputs = np.array([[75, 82, 93]]).T

X = data_inputs / np.amax(data_inputs, axis=0)
Y = data_outputs / 100

syn0 = 2*np.random.random((2, 1)) - 1
print(syn0.T)

for iter in range(0, 200):
    test_result = nonlin(np.dot(X, syn0))
    error = test_result - Y

    delta = error * nonlin(test_result, True)
    syn0 -= np.dot(X.T, delta)
    print(syn0.T)

print("Output After Training:")
print(test_result)



