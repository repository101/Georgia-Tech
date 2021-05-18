"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import pickle

import ps6_util
import ps6

# I/O directories
INPUT_DIR = "input_images"
OUTPUT_DIR = "./"

YALE_FACES_DIR = os.path.join(INPUT_DIR, 'Yalefaces')
FACES94_DIR = os.path.join(INPUT_DIR, 'faces94')
POS_DIR = os.path.join(INPUT_DIR, "pos")
NEG_DIR = os.path.join(INPUT_DIR, "neg")
NEG2_DIR = os.path.join(INPUT_DIR, "neg2")


def load_images_from_dir(data_dir, size=(24, 24), ext=".png"):
    imagesFiles = [f for f in os.listdir(data_dir) if f.endswith(ext)]
    imgs = [np.array(cv2.imread(os.path.join(data_dir, f), 0)) for f in imagesFiles]
    imgs = [cv2.resize(x, size) for x in imgs]

    return imgs

# Utility function
def plot_eigen_faces(eig_vecs, fig_name="", visualize=False):
    r = np.ceil(np.sqrt(len(eig_vecs)))
    c = int(np.ceil(len(eig_vecs)/r))
    r = int(r)
    fig = plt.figure()

    for i,v in enumerate(eig_vecs):
        sp = fig.add_subplot(r,c,i+1)
        plt.imshow(v.reshape(32,32).real, cmap='gray')
        sp.set_title('eigenface_%i'%i)
        sp.axis('off')

    fig.subplots_adjust(hspace=.5)

    if visualize:
        plt.show()

    if not fig_name == "":
        plt.savefig("{}".format(fig_name))


# Functions you need to complete
def visualize_mean_face(x_mean, size, new_dims):
    """Rearrange the data in the mean face to a 2D array

    - Organize the contents in the mean face vector to a 2D array.
    - Normalize this image.
    - Resize it to match the new dimensions parameter

    Args:
        x_mean (numpy.array): Mean face values.
        size (tuple): x_mean 2D dimensions
        new_dims (tuple): Output array dimensions

    Returns:
        numpy.array: Mean face uint8 2D array.
    """
    return cv2.resize(ps6.normalize_and_scale(np.reshape(x_mean, newshape=size)), dsize=new_dims)


def part_1a_1b():

    orig_size = (192, 231)
    small_size = (32, 32)
    X, y = ps6.load_images(YALE_FACES_DIR, small_size)

    # Get the mean face
    x_mean = ps6.get_mean_face(X)

    x_mean_image = visualize_mean_face(x_mean, small_size, orig_size)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "ps6-1-a-1.png"), x_mean_image)

    # PCA dimension reduction
    k = 10
    eig_vecs, eig_vals = ps6.pca(X, k)

    plot_eigen_faces(eig_vecs.T, "ps6-1-b-1.png")
    plt.close('all')


def part_1c():
    load_data = True
    store = pd.HDFStore("part_1c.h5")
    
    if not load_data:
        for run in range(5):
            index = np.round(np.arange(0.05, 0.96, 0.05), 2)  # Split Percent
            columns = np.arange(1, 21, 1)  # Number of Eigenvalues to take
            dataframe = pd.DataFrame(index=index, columns=columns, data=np.zeros(shape=(index.shape[0], columns.shape[0])))
        
            for ind in index:
                # Index is the split percentage
                for col in columns:
                    # Col is the number of eigenvalues to take
                    p = ind  # Select a split percentage value
                    k = col  # Select a value for k
                
                    size = (32, 32)
                    X, y = ps6.load_images(YALE_FACES_DIR, size)
                    Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)
                
                    # training
                    mu = ps6.get_mean_face(Xtrain)
                    eig_vecs, eig_vals = ps6.pca(Xtrain, k)
                    Xtrain_proj = np.dot(Xtrain - mu, eig_vecs)
                
                    # testing
                    mu = ps6.get_mean_face(Xtest)
                    Xtest_proj = np.dot(Xtest - mu, eig_vecs)
                
                    good = 0
                    bad = 0
                
                    for i, obs in enumerate(Xtest_proj):
                
                        dist = [np.linalg.norm(obs - x) for x in Xtrain_proj]
                
                        idx = np.argmin(dist)
                        y_pred = ytrain[idx]
                
                        if y_pred == ytest[i]:
                            good += 1
                
                        else:
                            bad += 1
                            
                    accuracy = 100 * float(good) / (good + bad)
                    dataframe.at[ind, col] = accuracy
        
                    print('Good predictions = ', good, 'Bad predictions = ', bad)
                    print('{0:.2f}% accuracy'.format(100 * float(good) / (good + bad)))
            store[f"{run}"] = dataframe
        store.close()
    else:

        df_0 = store["0"]
        df_1 = store["1"]
        df_2 = store["2"]
        df_3 = store["3"]
        df_4 = store["4"]
        avg_df = (df_0 + df_1 + df_2 + df_3 + df_4) / 5
        avg = avg_df.to_numpy()
        ps6_util.heatmap(avg[::2, ::2]/100, row_labels=np.round(np.arange(0.1, 0.91, 0.1), 2),
                         col_labels=np.arange(2, 19, 2), x_label="K", y_label="Split Percent",
                         title="Average Accuracy using Eigenfaces", filename="ps6_part1_c_heatmap")
        print("Finished Heatmap")
        plt.close('all')


def part_2a():
    load_data = True
    y0 = 1
    y1 = 2

    X, y = ps6.load_images(FACES94_DIR)

    # # Select only the y0 and y1 classes
    # idx = y == y0
    # idx |= y == y1
    #
    # X = X[idx,:]
    # y = y[idx]
    #
    # # Label them 1 and -1
    # y0_ids = y == y0
    # y1_ids = y == y1
    # y[y0_ids] = 1
    # y[y1_ids] = -1
    #
    # p = 0.8
    # Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)
    # # Picking random numbers
    # rand_y = np.random.choice([-1, 1], (len(ytrain)))
    # rand_accuracy = np.count_nonzero(rand_y == ytrain) / ytrain.shape[0]
    # print('(Random) Training accuracy: {0:.2f}%'.format(rand_accuracy))
    #
    # # Using Weak Classifier
    # uniform_weights = np.ones((Xtrain.shape[0],)) / Xtrain.shape[0]
    # wk_clf = ps6.WeakClassifier(Xtrain, ytrain, uniform_weights)
    # wk_clf.train()
    # wk_results = [wk_clf.predict(x) for x in Xtrain]
    # wk_accuracy = np.count_nonzero(wk_results == ytrain) / ytrain.shape[0]
    # print('(Weak) Training accuracy {0:.2f}%'.format(wk_accuracy))
    #
    # num_iter = 5
    #
    # boost = ps6.Boosting(Xtrain, ytrain, num_iter)
    # boost.train()
    # good, bad = boost.evaluate()
    # boost_accuracy = 100 * float(good) / (good + bad)
    # print('(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy))
    #
    # # Picking random numbers
    # rand_y = np.random.choice([-1, 1], (len(ytest)))
    # rand_accuracy = np.count_nonzero(rand_y == ytest) / ytest.shape[0]
    # print('(Random) Testing accuracy: {0:.2f}%'.format(rand_accuracy))
    #
    # # Using Weak Classifier
    # wk_results = [wk_clf.predict(x) for x in Xtest]
    # wk_accuracy = np.count_nonzero(wk_results == ytest) / ytest.shape[0]
    # print('(Weak) Testing accuracy {0:.2f}%'.format(wk_accuracy))
    #
    # y_pred = boost.predict(Xtest)
    # boost_accuracy = np.count_nonzero(y_pred == ytest) / ytest.shape[0]
    # print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy))
    # print()
    
    # Number of iterations
    # num_iterations = np.arange(1, 51, 1).astype(np.int)
    num_iterations = np.arange(5, 21, 5).astype(np.int)
    p_vals = np.arange(1, 10, 1).astype(np.int)
    store = pd.HDFStore("part_2a_pt2.h5")
    # t = store["2a"]
    if not load_data:
        cols = [f"boost_{i}" for i in num_iterations]
        cols.append("weak")
        cols.append("random")
        cols = np.asarray(cols)
        training_dataframe = pd.DataFrame(data=np.zeros(shape=(p_vals.shape[0], cols.shape[0])), columns=cols, index=p_vals)
        testing_dataframe = pd.DataFrame(data=np.zeros(shape=(p_vals.shape[0], cols.shape[0])), columns=cols, index=p_vals)
        
        for p_val in p_vals:
            print(f"Current P_val: {p_val/10.0}")
            X, y = ps6.load_images(FACES94_DIR)
        
            # Select only the y0 and y1 classes
            idx = y == y0
            idx |= y == y1
        
            X = X[idx, :]
            y = y[idx]
        
            # Label them 1 and -1
            y0_ids = y == y0
            y1_ids = y == y1
            y[y0_ids] = 1
            y[y1_ids] = -1
            
            p = p_val / 10.0
    
            Xtrain, ytrain, Xtest, ytest = ps6.split_dataset(X, y, p)
            # Picking random numbers
            rand_y_train = np.random.choice([-1, 1], (len(ytrain)))
            rand_accuracy_train = np.count_nonzero(rand_y_train == ytrain) / ytrain.shape[0]
            training_dataframe["random"].at[p_val] = rand_accuracy_train
            # print('(Random) Training accuracy: {0:.2f}%'.format(rand_accuracy))
    
            # Using Weak Classifier
            uniform_weights = np.ones((Xtrain.shape[0],)) / Xtrain.shape[0]
            wk_clf = ps6.WeakClassifier(Xtrain, ytrain, uniform_weights)
            wk_clf.train()
            wk_results_train = [wk_clf.predict(x) for x in Xtrain]
            wk_accuracy_train = np.count_nonzero(wk_results_train == ytrain) / ytrain.shape[0]
            training_dataframe["weak"].at[p_val] = wk_accuracy_train
            # print('(Weak) Training accuracy {0:.2f}%'.format(wk_accuracy))
    
            # Picking random numbers
            rand_y_test = np.random.choice([-1, 1], (len(ytest)))
            rand_accuracy_test = np.count_nonzero(rand_y_test == ytest) / ytest.shape[0]
            testing_dataframe["random"].at[p_val] = rand_accuracy_test
            # print('(Random) Testing accuracy: {0:.2f}%'.format(rand_accuracy))
    
            # Using Weak Classifier
            wk_results_test = [wk_clf.predict(x) for x in Xtest]
            wk_accuracy_test = np.count_nonzero(wk_results_test == ytest) / ytest.shape[0]
            testing_dataframe["weak"].at[p_val] = wk_accuracy_test
            # print('(Weak) Testing accuracy {0:.2f}%'.format(wk_accuracy))
            
            for iteration in num_iterations:
                boost = ps6.Boosting(Xtrain, ytrain, iteration)
                boost.train()
                good, bad = boost.evaluate()
                boost_accuracy = float(good) / (good + bad)
                # print(f"\nCurrent Iteration: {iteration}")
                # print('(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy))
                training_dataframe[f"boost_{iteration}"].at[p_val] = boost_accuracy
        
                y_pred_test = boost.predict(Xtest)
                boost_accuracy_test = np.count_nonzero(y_pred_test == ytest) / ytest.shape[0]
                testing_dataframe[f"boost_{iteration}"].at[p_val] = boost_accuracy_test
                # print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy))
        store["2a_train"] = training_dataframe
        store["2a_test"] = testing_dataframe
        store.close()
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(20,10))
    
    training_dataframe = store["2a_train"]
    testing_dataframe = store["2a_test"]
    # test_ax = training_dataframe["test"].plot(label="Testing Accuracy", legend=True)
    training_dataframe[f"random"].plot(label=f"random_training_accuracy",
                                          title="Training Accuracy", xlabel="P Values",
                                          ylabel="Accuracy", ax=ax1)
    ax1.set_xticklabels((p_vals/10.0))
    for i in num_iterations:
        training_dataframe[f"boost_{i}"].plot(label=f"boost_{i}_training_accuracy", ax=ax1)

    training_dataframe[f"weak"].plot(label=f"weak_learner_training_accuracy", ax=ax1)

    testing_dataframe[f"random"].plot(label=f"random_testing_accuracy",
                                                  title="Testing Accuracy", xlabel="P Values",
                                                  ylabel="Accuracy", ax=ax2)
    ax2.set_xticklabels((p_vals / 10.0))
    for i in num_iterations:
        testing_dataframe[f"boost_{i}"].plot(label=f"boost_{i}_testing_accuracy", ax=ax2)

    testing_dataframe[f"weak"].plot(label=f"weak_learner_testing_accuracy", ax=ax2)
    plt.legend(bbox_to_anchor=(-.6, 1.05, 1.05, .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0.)
    plt.savefig("part_2a_AccuracyVsPVal_Testing.png", bbox_inches='tight')
    #
    print()


def part_3a():
    """Complete the remaining parts of this section as instructed in the
    instructions document."""
    haar_val = {"two_horizontal": (2, 1), "two_vertical": (1, 2), "three_horizontal": (3, 1), "three_vertical": (1, 3),
                "four_square": (2, 2)}

    feature1 = ps6.HaarFeature(haar_val["two_horizontal"], (25, 30), (50, 100))
    feature1.preview((200, 200), filename="ps6-3-a-1.png")
    
    feature2 = ps6.HaarFeature(haar_val["two_vertical"], (10, 25), (50, 150))
    feature2.preview((200, 200), filename="ps6-3-a-2.png")
    
    feature3 = ps6.HaarFeature(haar_val["three_horizontal"], (50, 50), (100, 50))
    feature3.preview((200, 200), filename="ps6-3-a-3.png")
    
    feature4 = ps6.HaarFeature(haar_val["three_vertical"], (50, 125), (100, 50))
    feature4.preview((200, 200), filename="ps6-3-a-4.png")
    
    feature5 = ps6.HaarFeature(haar_val["four_square"], (50, 25), (100, 150))
    feature5.preview((200, 200), filename="ps6-3-a-5.png")


def part_4_a_b():
    get_time = False
    if get_time:
        import time
    pos = load_images_from_dir(POS_DIR)
    neg = load_images_from_dir(NEG_DIR)

    train_pos = pos[:25]
    train_neg = neg[:]
    images = train_pos + train_neg
    labels = np.array(len(train_pos) * [1] + len(train_neg) * [-1])

    integral_images = ps6.convert_images_to_integral_images(images)
    VJ = ps6.ViolaJones(train_pos, train_neg, integral_images)
    num_runs = 5
    avg_time = np.zeros(shape=(num_runs,))
    if get_time:
        for i in range(num_runs):
            start_time = time.time()
            VJ.createHaarFeatures()
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_time[i] = elapsed_time
            
        avg = np.mean(avg_time)
    else:
        VJ.createHaarFeatures()
    VJ.train(5)

    VJ.haarFeatures[VJ.classifiers[0].feature].preview(filename="ps6-4-b-1")
    VJ.haarFeatures[VJ.classifiers[1].feature].preview(filename="ps6-4-b-2")

    predictions = VJ.predict(images)
    vj_accuracy = np.count_nonzero(predictions == labels) / labels.shape[0]
    print("Prediction accuracy on training: {0:.2f}%".format(vj_accuracy))

    neg = load_images_from_dir(NEG2_DIR)

    test_pos = pos[25:]
    test_neg = neg[:35]
    test_images = test_pos + test_neg
    real_labels = np.array(len(test_pos) * [1] + len(test_neg) * [-1])
    predictions = VJ.predict(test_images)

    vj_accuracy = np.count_nonzero(predictions == real_labels) / real_labels.shape[0]
    print("Prediction accuracy on testing: {0:.2f}%".format(vj_accuracy))


def part_4_c():
    load_clf = False
    f_name = "ViolaJonesClassifierTrained.pkl"
    
    pos = load_images_from_dir(POS_DIR)[:20]
    neg = load_images_from_dir(NEG_DIR)

    images = pos + neg

    integral_images = ps6.convert_images_to_integral_images(images)
    if load_clf:
        print()
        with open(f_name, "rb") as input_file:
            VJ = pickle.load(input_file)
            
    else:
        VJ = ps6.ViolaJones(pos, neg, integral_images)
        VJ.createHaarFeatures()
    
        VJ.train(4)
        with open(f_name, "wb") as output_file:
            pickle.dump(VJ, output_file)
    

    image = cv2.imread(os.path.join(INPUT_DIR, "man.jpeg"), -1)
    image = cv2.resize(image, (120, 60))
    VJ.faceDetection(image, filename="ps6-4-c-1")


if __name__ == "__main__":
    # part_1a_1b()
    # part_1c()
    # part_2a()
    # part_3a()
    # part_4_a_b()
    part_4_c()
