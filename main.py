import os

from nolearn.decaf import ConvNetFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

DECAF_IMAGENET_DIR = '/path/to/imagenet-files/'
TRAIN_DATA_DIR = '~/Desktop/Cat&Dog/all/train/'

def get_dataset():
    cat_dir = TRAIN_DATA_DIR + 'cat/'
    cat_filenames = [cat_dir + fn for fn in os.listdir(cat_dir)]
    dog_dir = TRAIN_DATA_DIR + 'dog/'
    dog_filenames = [dog_dir + fn for fn in os.listdir(dog_dir)]

    labels = [0] * len(cat_filenames) + [1] * len(dog_filenames)
    filenames = cat_filenames + dog_filenames
    return shuffle(filenames, labels, random_state=0)


def main():
    convnet = ConvNetFeatures(
        pretrained_params=DECAF_IMAGENET_DIR + 'imagenet.decafnet.epoch90',
        pretrained_meta=DECAF_IMAGENET_DIR + 'imagenet.decafnet.meta',
        )
    clf = LogisticRegression()
    pl = Pipeline([
        ('convnet', convnet),
        ('clf', clf),
        ])

    X, y = get_dataset()
    X_train, y_train = X[:100], y[:100]
    X_test, y_test = X[100:300], y[100:300]

    print ("Fitting...")
    pl.fit(X_train, y_train)
    print ("Predicting...")
    y_pred = pl.predict(X_test)
    print ("Accuracy: %.3f" % accuracy_score(y_test, y_pred))

main()