import tensorflow as tf
import matplotlib.pyplot as plt

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

#print(trainX[0])
#print(trainX.shape)
#print(trainY)

#plt.imshow(trainX[3])
#plt.gray()
#plt.colorbar()
#plt.show()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(28, 28), activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(trainX, trainY, epochs=5)
