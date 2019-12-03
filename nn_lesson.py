import numpy as np

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))


data_inputs = np.array([[3, 5], [5, 1], [10, 2]])
data_outputs = np.array([[75, 82, 93]]).T

X = data_inputs / np.amax(data_inputs, axis=0)
Y = data_outputs / 100

# начальное задание синапсов
syn0 = 2*np.random.random((2, 1)) - 1
print('Начальное задание синапсов')
print(syn0.T)

lessons = 2000

for iter in range(0, lessons):
    test_result = nonlin(np.dot(X, syn0))
    error = test_result - Y

    delta = error * nonlin(test_result, True)
    syn0 -= np.dot(X.T, delta)

print("Output After Training:")
print('Итоговые значения синапсов')
print(syn0.T)
print('Достигнутый результат после ', lessons, ' уроков')
print(test_result)
print('Нужный результат')
print(Y)



