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

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


GRAY_THRESHOLD = 120
IMG_SIZE = 28
NUMBER_OF_IMAGES = 60000

TRAIN_SET_SIZE = 15000
TEST_SET_SIZE = 3000
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
Train_Images, Test_Images, Train_Labels, Test_Labels = train_test_split(Images, Labels, test_size=(TEST_SET_SIZE/SET_SIZE), random_state=32)

# Defining a weak learner
svc = SVC(probability=True, kernel='linear') # 90.7% slow
lr = LogisticRegression(random_state=1) # 87.2%
rfc = RandomForestClassifier(n_estimators=40, random_state=1, criterion='gini') # 95.13% fast
                # for smaller samples entropy works better
gnb = GaussianNB() # 64.4%

classifier = AdaBoostClassifier(n_estimators=40, base_estimator=rfc, learning_rate=1.2)
model = classifier.fit(Train_Images, Train_Labels)

y_pred = model.predict(Test_Images)

print(f'Accuracy: {metrics.accuracy_score(Test_Labels, y_pred) * 100} %')

# Plotting few images
tmp = []
for i in range(9):
    print(f'Label: {Test_Labels[i]} | Prediction: {y_pred[i]}')
    pyplot.subplot(330 + 1 + i)
    tmp = np.reshape(Test_Images[i], (IMG_SIZE, IMG_SIZE))
    pyplot.imshow(tmp)
pyplot.show()

#AI Project
# Pie no1
#results_df = pd.DataFrame({'Labels': Test_Labels,
#                           'Prediction': y_pred})
#
#s_list = []
#
#for i in range(9):
#    if y_pred[i] == Test_Labels[i]:
#        s_list.append('Success')
#    else:
#        s_list.append('Fail')
#
#results_df['Outcome'] = s_list
#results_df.plot.pie(y=s_list)
#
## Pie no2
#fig1, ax = pyplot.subplots()
#patches, text, auto = ax.pie(s_list, autopct='%1.1f%%', colors=['red','green'],
#                             wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
#                             pctdistance=1.15, startangle=160)
#circle = pyplot.Circle((0, 0), 0.5, color='white')
#pyplot.gcf().gca().add_artist(circle)
#pyplot.legend(patches, ['Success', 'Fail'], loc='upper right', fontsize='xx-small', framealpha=0.4)
