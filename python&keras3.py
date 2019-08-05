# importação de pacotes necessários
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# importar o MNIST
print("[INFO] importando MNIST...")
fashion_mnist = keras.datasets.fashion_mnist

# carrega o dataset entre train (60.0000 imagens) e test (10.0000 imagens)
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# inicializa a lista de rótulos de classes
class_names = ['Camiseta', 'Calça', 'Sueter', 'Vestido', 'Casaco',
               'Sandalia', 'Camisa', 'Tenis', 'Mochila', 'Bota']

# normalizar todos pixels, de forma que os valores estejam
# no intervalor [0, 1.0]
train_images = train_images / 255.0
test_images = test_images / 255.0

# definir a arquitetura da Rede Neural usando Keras
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# treinar o modelo usando Algoritmo de Adam
print("[INFO] treinando a rede neural...")
epoca = 5
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
H=model.fit(train_images, train_labels,batch_size=128 , epochs=epoca, verbose=2)

# avaliar a Rede Neural
print("[INFO] avaliando a rede neural...")
predictions = model.predict(test_images, batch_size=128)

# plotagem da imagem de teste
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

# plotagem do gráfico da imagem de teste
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# plotagem de todos os resultados
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)

# plotagem de resultado unitario
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

# plotagem de perda e precisao para o dataset 'train'
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epoca), H.history["loss"], label="treino_loss")
plt.plot(np.arange(0, epoca), H.history["acc"], label="treino_acc")
plt.title("Treino Perda e Precisao")
plt.xlabel("Epoca #")
plt.ylabel("Perda/Precisao")
plt.legend()

# exibir plotagens
plt.show()
