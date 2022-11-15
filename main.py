# Slutprojekt för Big Data
import warnings
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skimage.io import imread
from skimage.measure import find_contours
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score)

# Ignore the warning about f1_score not working with multiclass
warnings.filterwarnings('ignore')


nr_of_letters = 26  # There are 26 english letters


def prepare_img(filename):  # Function to prepare images
    img = imread(filename, as_gray=True)
    contours = find_contours(img, .9)

    xmin = img.shape[1]
    ymin = img.shape[0]
    xmax = ymax = 0
    for contour in contours:
        x_var = contour[:, 1]
        y_var = contour[:, 0]
        xmin = min(xmin, x_var.min())
        xmax = max(xmax, x_var.max())
        ymin = min(ymin, y_var.min())
        ymax = max(ymax, y_var.max())

    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax + .9)
    ymax = int(ymax + .9)

    cropped = img[ymin:ymax, xmin:xmax]
    resized = resize(cropped, (8, 8), mode='constant', cval=1)

    return resized


# Loop through and prepare all images
srcdir = 'abc/'
np.set_printoptions(precision=2)

abclist = []
for imfile in os.listdir(srcdir):
    if imfile[-4:].lower() != '.png':
        continue

    imgdata = prepare_img(srcdir + '/' + imfile)
    abcdata = np.floor(np.array(imgdata) * 100).astype(int).flatten()

    abclist.append([*abcdata, imfile[0], imfile])

columns = [f"px{i}{j}" for i in range(8) for j in range(8)] + ['abc', 'file']


df = pd.DataFrame(abclist, columns=columns)  # Create dataframe
X = df.iloc[:, :64]  # The first 64 columns are the pixels
y = df['abc']  # The abc column is the letter its supposed to be


# Group by letter and calculate the mean of each column
pix_mean = X.groupby(y).mean()

fig, ax = plt.subplots(nr_of_letters)

# show all the letters in a plot
for i in range(nr_of_letters):
    ax[i].imshow(pix_mean.iloc[i, :].values.reshape(8, 8), cmap='gray')
    ax[i].set_title(i)
    ax[i].set_yticks(range(8))
    ax[i].set_xticks(range(8))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=1)  # Split data into train and test


# Define all models
models = [DecisionTreeClassifier(), SVC(), SVC(kernel='linear'), KNeighborsClassifier(
), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), MLPClassifier()]

# Loop through all models and print the accuracy
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Model: ", model, "Classification report:\n",
          classification_report(y_test, y_pred))
    print("Model: ", model, "Confusion matrix:\n",
          confusion_matrix(y_test, y_pred))
    print("Model:", model, "Accuracy:", accuracy_score(y_test, y_pred))

# plt.show()
