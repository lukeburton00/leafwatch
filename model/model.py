import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

def main():
    # data paths
    train_data_path = "../dataset/train"
    val_data_path = "../dataset/validate"
    data_folders = os.listdir(train_data_path)

    # training constants
    EPOCHS: int = 25
    BATCH_SIZE: int = 32
    WIDTH: int = 128
    HEIGHT: int = 128

    # data augmentation
    generator = ImageDataGenerator(
        rescale = 1.0/255,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        brightness_range=[0.4, 1.4],
        channel_shift_range=15,
    )

    training_data = generator.flow_from_directory(
        train_data_path,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True,
        classes=[name for name in data_folders]
    )

    val_data = generator.flow_from_directory(
        val_data_path,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True,
        classes=[name for name in data_folders]
    )

    # convolutional neural network definition
    model = models.Sequential()

    model.add(layers.Conv2D(filters=64, kernel_size=9, padding='same', strides=(2, 2), activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(filters=128, kernel_size=9, strides=(1, 1), activation='relu'))
    model.add(layers.Conv2D(filters=256, kernel_size=9, strides=(1, 1), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(units=1024, activation='relu'))
    model.add(layers.Dropout(0.4))   
    model.add(layers.Dense(units=64, activation='softmax'))

    model.summary()

    # model compilation and training
    model.compile(
        optimizer='adam', 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy'])
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',  
        patience=5,          
        restore_best_weights=True
    )

    history = model.fit(
        training_data,
        epochs=EPOCHS,
        validation_data=val_data,
        # callbacks=[early_stopping]
    )

    # display training accuracy data
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    # serialize model
    should_continue_str = input("Would you like to save the model? (y/n)")

    if (should_continue_str == "y"):
        model.save("plant_disease_classification.h5")
    else:
        quit()

if __name__ == "__main__":
    main()