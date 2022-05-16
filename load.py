from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
# Load libraries
from sklearn.ensemble import AdaBoostClassifier
# Import Support Vector Classifier
from sklearn.svm import SVC
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import numpy as np
import pandas as pd

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


def create_pie_chart(Test_Labels, y_pred):
    status_list = []

    for i in range(len(Test_Labels)):
        if y_pred[i] == Test_Labels[i]:
            status_list.append('Success')
        else:
            status_list.append('Fail')

    proportion = [status_list.count('Success'), status_list.count('Fail')]

    # Pie no2
    fig1, ax = pyplot.subplots()
    patches, text, auto = ax.pie(proportion, autopct='%1.1f%%', colors=['#43C456', '#FF6565'],
                                 wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
                                 pctdistance=1.15, startangle=160)
    circle = pyplot.Circle((0, 0), 0.5, color='white')
    pyplot.gcf().gca().add_artist(circle)
    pyplot.legend(patches, ['Success', 'Fail'], loc='upper right', fontsize='xx-small', framealpha=0.4)

    pyplot.show()


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
Train_Images, Test_Images, Train_Labels, Test_Labels = train_test_split(Images, Labels, test_size=0.2, random_state=32)

# Defining a weak learner
svc = SVC(probability=True, kernel='linear')

# Creating and training a classifier
classifier = AdaBoostClassifier(n_estimators=30, base_estimator=svc, learning_rate=1)
model = classifier.fit(Train_Images, Train_Labels)

y_pred = model.predict(Test_Images)

print("Accuracy:", metrics.accuracy_score(Test_Labels, y_pred))

# Plotting few images
tmp = []
for i in range(9):
    print(f'Label: {Test_Labels[i]} | Prediction: {y_pred[i]}')
    pyplot.subplot(330 + 1 + i)
    tmp = np.reshape(Test_Images[i], (IMG_SIZE, IMG_SIZE))
    pyplot.imshow(tmp)
pyplot.show()

create_pie_chart(Test_Labels, y_pred)
