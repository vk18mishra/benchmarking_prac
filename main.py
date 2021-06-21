# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
import pprofile
import os
import tensorflow as tf
#from keras import backend as K
#from pyJoules.device.rapl_device import RaplPackageDomain
from tensorflow.keras.preprocessing import image_dataset_from_directory
#from model_profiler import model_profiler
#from memory_profiler import profile
# from line_profiler import LineProfiler
import cProfile, pstats, sys
import io
import pyRAPL
import pandas as pd
import psutil
#from pyJoules.energy_meter import measure_energy
#from pyJoules.handler.csv_handler import CSVHandler
from tabulate import tabulate

data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])

def show_images(train_dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

def show_images_new(train_dataset):
    for image, _ in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        first_image = image[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
            plt.imshow(augmented_image[0] / 255)
            plt.axis('off')

pyRAPL.setup()

csv_output = pyRAPL.outputs.CSVOutput('energy_pyRAPL.csv')

@pyRAPL.measure(output=csv_output)
def train(model, train_dataset, validation_dataset, initial_epochs):
    history = model.fit(train_dataset, epochs=initial_epochs, validation_data=validation_dataset)
    return history

def preprocessing():
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    _URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
    path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
    PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')

    BATCH_SIZE = 32
    IMG_SIZE = (160, 160)

    train_dataset = image_dataset_from_directory(train_dir,
                                                 shuffle=True,
                                                 batch_size=BATCH_SIZE,
                                                 image_size=IMG_SIZE)

    validation_dataset = image_dataset_from_directory(validation_dir,
                                                      shuffle=True,
                                                      batch_size=BATCH_SIZE,
                                                      image_size=IMG_SIZE)

    class_names = train_dataset.class_names

    show_images(train_dataset, class_names)

    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
    print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    #     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    # ])

    show_images_new(train_dataset)

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1)

    #K.clear_session()

    # Create the base model from the pre-trained model MobileNet V2
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')

    image_batch, label_batch = next(iter(train_dataset))
    feature_batch = base_model(image_batch)
    print(feature_batch.shape)

    base_model.trainable = False

    # Let's take a look at the base model architecture
    print(base_model.summary())

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)

    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)

    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())

    print(len(model.trainable_variables))

    initial_epochs = 2

    loss0, accuracy0 = model.evaluate(validation_dataset)

    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    # @profile
    # def train(model):
    #     history = model.fit(train_dataset,
    #                         epochs=initial_epochs,
    #                         validation_data=validation_dataset)
    #     return history

    # gives a single float value
    cpu_per_b = psutil.cpu_percent()
    # gives an object with many fields
    # vir_mem_b = psutil.virtual_memory()
    # you can convert that object to a dictionary
    vir_mem_b = dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    vir_mem_per_b = psutil.virtual_memory().percent
    # you can calculate percentage of available memory
    mem_av_per_b = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

    profiler = pprofile.Profile()
    pr = cProfile.Profile()
    pr.enable()

    with profiler:
      # csv_handler = CSVHandler('energy_pyjoules.csv')

      history = train(model, train_dataset, validation_dataset, initial_epochs)
      csv_output.save()

      # gives a single float value
    profiler.print_stats()
    # Or to a file:
    profiler.dump_stats("exec_time.txt")
    cpu_per_a = psutil.cpu_percent()
    # gives an object with many fields
    # vir_mem_a = psutil.virtual_memory()
    # you can convert that object to a dictionary
    vir_mem_a = dict(psutil.virtual_memory()._asdict())
    # you can have the percentage of used RAM
    vir_mem_per_a = psutil.virtual_memory().percent
    # you can calculate percentage of available memory
    mem_av_per_a = psutil.virtual_memory().available * 100 / psutil.virtual_memory().total

    pr.disable()
    ps = pstats.Stats(pr, stream=sys.stdout)
    ps.print_stats()

    CPU_table = [["CPU Usage Percent", cpu_per_b, cpu_per_a], ["Memory Used", vir_mem_b, vir_mem_a],
             ["Memory Used(%)", vir_mem_per_b, vir_mem_per_a], ["Memory Available(%)", mem_av_per_b, mem_av_per_a]]
    print(tabulate(CPU_table, headers=["Metric", "Before Training", "After Training"], tablefmt="pretty"))
    with open('CPU_table.txt', 'w') as f:
        f.write(tabulate(CPU_table, headers=["Metric", "Before Training", "After Training"], tablefmt="pretty"))
    f.close()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    #ps.print_stats()

    with open('memory_logs.txt', 'w+') as f:
        f.write(s.getvalue())
    f.close()

    # data_cpu = pd.read_csv("energy_pyRAPL.csv")
    # Preview the first 5 lines of the loaded data
    # data_cpu.head()


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

    # profile = model_profiler(model, BATCH_SIZE)

    # print(profile)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    preprocessing()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
