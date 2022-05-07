from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import numpy as np

GRAY_THRESHOLD = 120
IMG_SIZE = 28
NUMBER_OF_IMAGES = 60000

TRAIN_SET_SIZE = 4000
TEST_SET_SIZE = 1000
SET_SIZE = TRAIN_SET_SIZE + TEST_SET_SIZE

def load_data():
    Images, Labels = loadlocal_mnist(
            images_path='train-images.idx3-ubyte', 
            labels_path='train-labels.idx1-ubyte')
    return Images, Labels

def remove_gray(Image):
    for i in range(784):
        if Image[i] < GRAY_THRESHOLD:
            Image[i] = 0
        else:
            Image[i] = 255
    return Image

# Loading all data
ImagesAll, LabelsAll = load_data()

# Taking a part of data
# I'm doing it because processing 60 000 of images takes too long
Images = ImagesAll[:SET_SIZE]
Labels = LabelsAll[:SET_SIZE]

# Removing gray pixels
for i in range(SET_SIZE):
    Images[i] = remove_gray(Images[i])

# Spliting choosen part of a data into train and test set
Train_Images, Test_Images, Train_Labels, Test_Labels = train_test_split(Images, Labels, test_size=0.2, random_state=42)

# Plotting few images
tmp = []
for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    tmp = np.reshape(Train_Images[i], (IMG_SIZE, IMG_SIZE))
    pyplot.imshow(tmp)
print(tmp)
pyplot.show()
