from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input
from keras.optimizers import SGD, Adadelta
import numpy as np

input_posit = Input(shape=(2,))

expanded = Dense(4, activation='relu')(input_posit)
expanded = Dense(8, activation='relu')(expanded)
expanded = Dense(16, activation='relu')(expanded)

compressed = Dense(8, activation='relu')(expanded)
compressed = Dense(4, activation='relu')(compressed)
compressed = Dense(2, activation='sigmoid')(compressed)

expander = Model(input_posit, expanded)

encoder = Model(input_posit, compressed)
 
x_data = np.random.rand(60000,2)
x_data.astype('float32')
x_data.shape

y_data = np.random.rand(10000,2) 
y_data.astype('float32')

encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

encoder.fit(x_data, x_data, epochs=50, batch_size=128, shuffle=True, validation_data=(y_data, y_data))

test = np.random.rand(10,2)
pred = encoder.predict(test)
diff = 100*(pred-test)/test
mean = abs(diff).mean()
garbage = expander.predict(test)

