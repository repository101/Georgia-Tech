"""Problem Set 6: PCA, Boosting, Haar Features, Viola-Jones."""
import numpy as np
import cv2
import os
import sys

from helper_classes import WeakClassifier, VJ_Classifier


# assignment code
def load_images(folder, size=(32, 32), load_color=False):
    """Load images to workspace.

    Args:
        folder (String): path to folder with images.
        size   (tuple): new image sizes

    Returns:
        tuple: two-element tuple containing:
            X (numpy.array): data matrix of flatten images
                             (row:observations, col:features) (float).
            y (numpy.array): 1D array of labels (int).
    """
    if load_color:
        dim_1 = (size[0] * size[1]) * 3
    else:
        dim_1 = (size[0] * size[1])
    images_files = [f for f in os.listdir(folder) if f.endswith(".png")]
    data = np.zeros(shape=(len(images_files), dim_1+1))
    for i in range(len(images_files)):
        temp_label = images_files[i].split(".")
        data[i, -1] = int(temp_label[0][-2:])
        if load_color:
            data[i, :-1] = cv2.resize(cv2.imread(f"{folder}\\" + images_files[i]), dsize=size).flatten()
        else:
            data[i, :-1] = cv2.resize(cv2.imread(f"{folder}\\"+images_files[i], 0), dsize=size).flatten()

    return data[:, :-1], data[:, -1]


def split_dataset(X, y, p, is_boosting=False):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    all_idx = np.arange(X.shape[0])
    perm = np.random.permutation(all_idx)
    idx = int(X.shape[0] * p)
    if is_boosting:
        y[y == -1] = 0
        return X[perm[:idx]], y[perm[:idx]], X[perm[idx:]], y[perm[idx:]]
    else:
        return X[perm[:idx]], y[perm[:idx]], X[perm[idx:]], y[perm[idx:]]


def get_mean_face(x):
    """Return the mean face.

    Calculate the mean for each column.

    Args:
        x (numpy.array): array of flattened images.

    Returns:
        numpy.array: Mean face.
    """
    return np.mean(x, axis=0)


# Utility function from PS4
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].
    
    Utility function. There is no need to modify it.
    
    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to [0, 255].
        
    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)
    
    return image_out

def pca(X, k):
    """PCA Reduction method.

    Return the top k eigenvectors and eigenvalues using the covariance array
    obtained from X.


    Args:
        X (numpy.array): 2D data array of flatten images (row:observations,
                         col:features) (float).
        k (int): new dimension space

    Returns:
        tuple: two-element tuple containing
            eigenvectors (numpy.array): 2D array with the top k eigenvectors.
            eigenvalues (numpy.array): array with the top k eigenvalues.
    """
    avg_face = get_mean_face(X)
    eig = np.linalg.eigh((X - avg_face).T.dot(X - avg_face))
    return eig[1][:, -k:][::, ::-1], eig[0][-k:][::-1]


class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.0001
        
    def calc_eps(self, pred_y):
        return np.sum(self.weights[pred_y != self.ytrain])

    def calc_alpha(self, eps):
        return 0.5 * np.log(((1 - eps) / eps))
    
    @staticmethod
    def normalize_weights(weights):
        return weights / np.sum(weights)

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for iteration in range(self.num_iterations):
            self.weights = self.normalize_weights(self.weights)
            clf = WeakClassifier(X=self.Xtrain, y=self.ytrain, weights=self.weights)
            clf.train()
            self.weakClassifiers.append(clf)
            pred_y = np.asarray([clf.predict(x=self.Xtrain[i]) for i in range(self.Xtrain.shape[0])])
            eps = self.calc_eps(pred_y=pred_y)
            self.alphas.append(self.calc_alpha(eps=eps))
            if eps < self.eps:
                break
            else:
                self.weights = np.multiply(self.weights, np.exp(-(np.multiply(self.ytrain, self.alphas[-1] * pred_y))))
        self.alphas = np.asarray(self.alphas)
        return

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        pred_y_correct = np.count_nonzero(self.predict(self.Xtrain) == self.ytrain)
        return pred_y_correct, self.ytrain.shape[0] - pred_y_correct

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        results = None
        for i in range(len(self.weakClassifiers)):
            if results is None:
                results = np.asarray([self.weakClassifiers[i].predict(X[j]) for j in range(len(X))])
            else:
                results = np.vstack((results, np.asarray([self.weakClassifiers[i].predict(X[j]) for j in range(len(X))])))
        # results = np.asarray([self.weakClassifiers[i].predict(X) for i in range(len(self.weakClassifiers))])
        
        return np.sign(np.sum(self.alphas * results.T, axis=1))
        

class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size
        
    def place_feature(self, shape, feature):
        temp_img = np.zeros(shape=shape)
        temp_img[self.position[0]:self.position[0] + self.size[0], self.position[1]: self.position[1] + self.size[1]] = feature
        return temp_img

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        temp_feature = np.full(shape=self.size, fill_value=126)
        temp_feature[:self.size[0] // 2, :self.size[1]] = 255
        return self.place_feature(shape=shape, feature=temp_feature)

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        temp_feature = np.full(shape=self.size, fill_value=126)
        temp_feature[:self.size[0], :self.size[1] // 2] = 255
        return self.place_feature(shape=shape, feature=temp_feature)

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        temp_feature = np.full(shape=self.size, fill_value=126)
        temp_feature[:self.size[0] // 3, :self.size[1]] = 255
        temp_feature[(self.size[0] // 3) * 2:, :self.size[1]] = 255
        return self.place_feature(shape=shape, feature=temp_feature)

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """

        temp_feature = np.full(shape=self.size, fill_value=126)
        temp_feature[:self.size[0], :self.size[1] // 3] = 255
        temp_feature[:self.size[0], (self.size[1] // 3) * 2:] = 255
        return self.place_feature(shape=shape, feature=temp_feature)

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        temp_feature = np.full(shape=self.size, fill_value=126)
        temp_feature[self.size[0] // 2:, :self.size[1] // 2] = 255
        temp_feature[:self.size[0] // 2, self.size[1] // 2:] = 255
        self.bottom_right = [(self.position[0])]
        return self.place_feature(shape=shape, feature=temp_feature)

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite("output/{}_feature.png".format(self.feat_type), X)

        else:
            cv2.imwrite("output/{}.png".format(filename), X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        get_times = False
        
        ii = ii.astype(np.float)

        feat_type = self.feat_type  # Change feature type
        pos = self.position
        size = self.size

        if feat_type == (2, 1):
            
            x1 = pos[0] - 1
            y1 = pos[1] - 1
            x2 = pos[0] + (size[0] // 2) - 1
            y2 = pos[1] + size[1] - 1
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            A = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1 + size[0] // 2
            y1 = pos[1] - 1
            x2 = pos[0] + size[0] - 1
            y2 = pos[1] + size[1] - 1
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            B = ptd - ptc - ptb + pta
            return A - B

        if feat_type == (1, 2):
            x1 = pos[0] - 1
            y1 = pos[1] - 1
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + size[1] // 2
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            A = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1
            y1 = pos[1] - 1 + size[1] // 2
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + size[1]
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            B = ptd - ptc - ptb + pta
            return A - B

        if feat_type == (3, 1):
            x1 = pos[0] - 1
            y1 = pos[1] - 1
            x2 = pos[0] - 1 + size[0] // 3
            y2 = pos[1] - 1 + size[1]
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            A = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1 + size[0] // 3
            y1 = pos[1] - 1
            x2 = pos[0] - 1 + 2 * size[0] // 3
            y2 = pos[1] - 1 + size[1]
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            B = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1 + 2 * size[0] // 3
            y1 = pos[1] - 1
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + size[1]
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            C = ptd - ptc - ptb + pta
            return A - B + C

        if feat_type == (1, 3):
            x1 = pos[0] - 1
            y1 = pos[1] - 1
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + size[1] // 3
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            A = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1
            y1 = pos[1] - 1 + size[1] // 3
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + 2 * size[1] // 3
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            B = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1
            y1 = pos[1] - 1 + 2 * size[1] // 3
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + size[1]
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            C = ptd - ptc - ptb + pta
            return A - B + C

        if feat_type == (2, 2):
            x1 = pos[0] - 1
            y1 = pos[1] - 1
            x2 = pos[0] - 1 + size[0] // 2
            y2 = pos[1] - 1 + size[1] // 2
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            A = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1
            y1 = pos[1] - 1 + size[1] // 2
            x2 = pos[0] - 1 + size[0] // 2
            y2 = pos[1] - 1 + size[1]
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            B = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1 + size[0] // 2
            y1 = pos[1] - 1
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + size[1] // 2
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            C = ptd - ptc - ptb + pta
            
            x1 = pos[0] - 1 + size[0] // 2
            y1 = pos[1] - 1 + size[1] // 2
            x2 = pos[0] - 1 + size[0]
            y2 = pos[1] - 1 + size[1]
            pta = ii[x1, y1]
            ptb = ii[x1, y2]
            ptc = ii[x2, y1]
            ptd = ii[x2, y2]
            D = ptd - ptc - ptb + pta
            return -A + B + C - D



def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    return [np.cumsum(np.cumsum(i, axis=0), axis=1) for i in images]


class ViolaJones:
    """Viola Jones face detection method

    Args:
        pos (list): List of positive images.
        neg (list): List of negative images.
        integral_images (list): List of integral images.

    Attributes:
        haarFeatures (list): List of haarFeature objects.
        integralImages (list): List of integral images.
        classifiers (list): List of weak classifiers (VJ_Classifier).
        alphas (list): Alpha values, one for each weak classifier.
        posImages (list): List of positive images.
        negImages (list): List of negative images.
        labels (numpy.array): Positive and negative labels.
    """
    def __init__(self, pos, neg, integral_images, use_sum=False):
        self.haarFeatures = []
        self.integralImages = integral_images
        self.classifiers = []
        self.alphas = []
        self.posImages = pos
        self.negImages = neg
        self.labels = np.hstack((np.ones(len(pos)), -1*np.ones(len(neg))))
        self.use_negative = True
        self.use_sum = use_sum

    def createHaarFeatures(self):
        # Let's take detector resolution of 24x24 like in the paper
        FeatureTypes = {"two_horizontal": (2, 1),
                        "two_vertical": (1, 2),
                        "three_horizontal": (3, 1),
                        "three_vertical": (1, 3),
                        "four_square": (2, 2)}

        haarFeatures = []
        for _, feat_type in FeatureTypes.items():
            for sizei in range(feat_type[0], 24 + 1, feat_type[0]):
                for sizej in range(feat_type[1], 24 + 1, feat_type[1]):
                    for posi in range(0, 24 - sizei + 1, 4):
                        for posj in range(0, 24 - sizej + 1, 4):
                            haarFeatures.append(
                                HaarFeature(feat_type, [posi, posj],
                                            [sizei, sizej]))
        self.haarFeatures = haarFeatures

    def train(self, num_classifiers):

        # Use this scores array to train a weak classifier using VJ_Classifier
        # in the for loop below.
        scores = np.zeros((len(self.integralImages), len(self.haarFeatures)))
        print(" -- compute all scores --")
        for i, im in enumerate(self.integralImages):
            scores[i, :] = [hf.evaluate(im) for hf in self.haarFeatures]

        weights_pos = np.ones(len(self.posImages), dtype='float') * 1.0 / (
                           2*len(self.posImages))
        weights_neg = np.ones(len(self.negImages), dtype='float') * 1.0 / (
                           2*len(self.negImages))
        weights = np.hstack((weights_pos, weights_neg))

        print(" -- select classifiers --")
        
        for i in range(num_classifiers):
            # Normalize Weights
            weights = weights / np.sum(weights)

            # Initialize Classifier h_j
            clf = VJ_Classifier(X=scores, y=self.labels, weights=weights)

            # Train Classifier
            clf.train()

            # Append classifier to self.classifiers
            self.classifiers.append(clf)
            
            # Calculate Error
            beta_t = clf.error / (1.0 - clf.error)

            # Update Weights
            if self.use_negative:
                e_i = np.asarray([-1 if clf.predict(scores[i]) == self.labels[i] else 1 for i in range(len(scores))])
            else:
                e_i = np.asarray([0 if clf.predict(scores[i]) == self.labels[i] else 1 for i in range(len(scores))])
            weights = weights * (beta_t ** (1 - e_i))

            # Calculate Alpha_t
            alpha_t = np.log((1.0 / beta_t))
            self.alphas.append(alpha_t)

        print(" -- training complete --")

    def predict(self, images):
        """Return predictions for a given list of images.

        Args:
            images (list of element of type numpy.array): list of images (observations).

        Returns:
            list: Predictions, one for each element in images.
        """

        integral_images = convert_images_to_integral_images(images)

        scores = np.zeros((len(integral_images), len(self.haarFeatures)))
        
        for i in range(len(self.classifiers)):
            # Populate the score location for each classifier 'clf' in
            # self.classifiers.
    
            # Obtain the Haar feature id from clf.feature
            feature_id = self.classifiers[i].feature
            for j in range(len(integral_images)):
                # Use this id to select the respective feature object from
                # self.haarFeatures
                # Add the score value to score[x, feature id] calling the feature's
                # evaluate function. 'x' is each image in 'ii'
                scores[j, feature_id] = self.haarFeatures[feature_id].evaluate(integral_images[j])

        result = []

        # Append the results for each row in 'scores'. This value is obtained
        # using the equation for the strong classifier H(x).
        total_alpha = np.sum(self.alphas)
        
        for x in scores:
            predictions_total = 0.
            for j in range(len(self.classifiers)):
                predictions_total += self.alphas[j] * self.classifiers[j].predict(x)
            if predictions_total >= total_alpha * 0.5:
                result.append(1)
            else:
                if self.use_negative:
                    result.append(-1)
                else:
                    result.append(0)
        return result

    def faceDetection(self, image, filename):
        """Scans for faces in a given image.

        Complete this function following the instructions in the problem set
        document.

        Use this function to also save the output image.

        Args:
            image (numpy.array): Input image.
            filename (str): Output image file name.

        Returns:
            None.
        """
        original_image = np.copy(image)
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape[:2]
        rectangle_dimension = 24

        windows = []
        index_tracker = []
        for temp_height in np.arange(0, height-rectangle_dimension, 1).astype(np.int):
            for temp_width in np.arange(0, width-rectangle_dimension, 1).astype(np.int):
                windows.append(image[temp_height:temp_height+24, temp_width:temp_width+24])
                index_tracker.append([temp_height, temp_width])
        pred_y = np.asarray(self.predict(windows))
        index_tracker = np.asarray(index_tracker)
        idx = pred_y == 1
        points = index_tracker[idx]
        temp_x = points[:, 0].mean().astype(np.int)
        temp_y = points[:, 1].mean().astype(np.int)
        offset = 6
        cv2.rectangle(original_image, pt1=(temp_y-offset, temp_x-offset), pt2=(temp_y+rectangle_dimension-offset, temp_x + rectangle_dimension-offset),
                      color=(0, 225, 0), thickness=1)
        
        cv2.imwrite(filename=f"output/{filename}.png", img=original_image)
        print()
