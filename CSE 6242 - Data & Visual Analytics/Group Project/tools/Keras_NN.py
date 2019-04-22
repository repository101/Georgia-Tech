import pandas as pd
import numpy as np

from keras.models import Sequential
from keras import layers
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(layers.Dense(8, input_dim=98, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def Neural_Network(file_name):
    the_dataframe = pd.read_csv(file_name, sep=",", header=0)
    #the_dataframe.drop('Selected', inplace=True, axis=1)
    print("X has been initialized: \n")
    scalar = StandardScaler()
    encoder = LabelEncoder()

    X = the_dataframe.drop("pitch_type", axis=1)
    # for col in X:
    #     print(X[col].unique())
    print("y has been initialized: \n")
    y = encoder.fit_transform(the_dataframe["pitch_type"])
    #y = scalar.fit_transform(y)
    # encoded_Y = encoder.transform(y)
    #
    # dummy_y = np_utils.to_categorical((encoded_Y))
    #
    # estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=200, verbose=1)
    # kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    print(len(list(the_dataframe.pitch_type.unique())))

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = Sequential()
    model.add(layers.Dense(20, input_dim=89, activation="relu", kernel_initializer="random_normal"))

    model.add(layers.Dense(20, activation="sigmoid", kernel_initializer="random_normal"))
    model.add(layers.Dense(1, activation="relu", kernel_initializer="random_normal"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=100, batch_size=50)

    model.summary()

    scores = model.evaluate(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)

    con_matrix = confusion_matrix(y_test, y_pred)
    #
    # print(con_matrix)

    # results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
    # print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    return


if __name__ == "__main__":
    file = "Data_New/434378.csv"
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    Neural_Network(file)
