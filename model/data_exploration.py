import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def main():
    # data paths
    train_data_path = "../dataset/train"
    data_folders = os.listdir(train_data_path)

    # category distribution data
    categories = data_folders
    category_counts = []

    # healthy vs diseased data
    healthy_count = 0
    diseased_count = 0

    # color distribution data
    color_dist = []
    channels = ['red', 'green', 'blue']

    num_images = 0
    # data extraction
    for folder in data_folders:
        category_counts.append(len(os.listdir(os.path.join(train_data_path, folder))))

        for image in os.listdir(os.path.join(train_data_path, folder)):
            img = cv2.imread(os.path.join(train_data_path, folder + "/" + image))

            b, g, r = cv2.split(img)
            dist = [np.mean(r), np.mean(g), np.mean(b)]
            color_dist.append(dist)

            if "healthy" in folder:
                healthy_count += 1
            else:
                diseased_count += 1
            
            num_images += 1

        print("Folder " + folder + " data extraction complete.")

    fig, ax = plt.subplots()
    channel_dist = np.mean(color_dist, axis=0)
    print(channel_dist)
    # display mean color distribution
    ax.bar(channels, channel_dist)
    plt.xticks(rotation=30, ha='right')
    ax.set_title('Channel distribution')

    plt.show()
    # display category distribution
    fig, ax = plt.subplots()

    ax.bar(categories, category_counts)
    plt.xticks(rotation=30, ha='right')

    ax.set_ylabel('Number of Images')
    ax.set_xlabel('Category')
    ax.set_title('Training Data Distribution')

    plt.show()

    labels = 'Healthy plants', 'Diseased plants'
    sizes = [healthy_count, diseased_count]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')

    plt.show()

if __name__ == "__main__":
    main()