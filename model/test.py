from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def main():
    test_data_path = "../dataset/test"
    data_folders = os.listdir(test_data_path)

    WIDTH = 128
    HEIGHT = 128
    BATCH_SIZE = 1

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
        test_data_path,
        target_size=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=True,
        classes=[name for name in data_folders]
    )

    model = models.load_model('plant_disease_classification.h5')
    score = model.evaluate(x=training_data)
    print(score)

if __name__ == "__main__":
    main()
