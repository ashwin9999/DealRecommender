from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from sklearn.preprocessing import StandardScaler
import numpy

numpy.random.seed(7)

dataset = numpy.loadtxt("../data/filled_ratings.csv", delimiter=",")
print(dataset.shape)

scalerX = StandardScaler()
scalerY = StandardScaler()

X = dataset[:,0:int(dataset.shape[1]/2)]
Y = dataset[:,int(dataset.shape[1]/2):]

scalerX.fit(X)
scalerY.fit(Y)

X = scalerX.transform(X)
Y = scalerY.transform(Y)

print('X.shape: {}'.format(X.shape))
print('Y.shape: {}'.format(Y.shape))

model = Sequential()
model.add(Dense(800, input_dim=25, kernel_initializer='normal', activation='relu'))
model.add(Dense(1200, kernel_initializer='normal', activation='relu'))
model.add(Dense(800, kernel_initializer='normal', activation='relu'))
model.add(Dense(25, kernel_initializer='normal'))

model.compile(loss='mean_absolute_error', optimizer='adam')
model.fit(X, Y, epochs=10000, batch_size=25)
test_output = model.predict(X)

to_2 = test_output
test_output = scalerX.inverse_transform(test_output)
Y = scalerY.inverse_transform(Y)
X = scalerX.inverse_transform(X)

numpy.savetxt("../data/nn_input.csv", X, delimiter=",")
numpy.savetxt("../data/nn_groundtruth.csv", Y, delimiter=",")
numpy.savetxt("../data/nn_output.csv", test_output, delimiter=",")
numpy.savetxt("../data/nn_output_normalized.csv", to_2, delimiter=",")
