
import math

def get_entropy(positive_examples, negative_examples):
    total = positive_examples + negative_examples
    return -((positive_examples/total) * math.log2((positive_examples/total))) - \
           ((negative_examples/total) * math.log2((negative_examples/total)))

def get_information_gain(positive_examples, positive_weak_examples, positive_strong_examples,
                         negative_examples, negative_weak_examples, negative_strong_examples):
    total = positive_examples + negative_examples
    total_weak = positive_weak_examples + negative_weak_examples
    total_strong = positive_strong_examples + negative_strong_examples
    total_entropy = get_entropy(positive_examples, negative_examples)
    weak_entropy = get_entropy(positive_weak_examples, negative_weak_examples)
    strong_entropy = get_entropy(positive_strong_examples, negative_strong_examples)

    return total_entropy - ((total_weak/total) * weak_entropy) - ((total_strong/total) * strong_entropy)


def get_true_positives(y, y_pred):
    np.zeros(shape=(y.shape[0],), dtype=bool)
    results = np.where(y == y_pred)
    return


def get_false_positives(y, y_pred):
    return


def get_true_negatives(y, y_pred):
    return


def get_false_negatives(y, y_pred):
    return


def calculate_precision(y, y_pred):
    """
            TruePositive / (TruePositive + FalsePositives)
    """
    tp = get_true_positives(y, y_pred)
    fp = get_false_positives(y, y_pred)
    return tp / (tp + fp)


def calculate_recall():
    """
            TruePositives / (TruePositives + FalseNegatives)
    """
    tp = get_true_positives()
    fn = get_false_negatives()
    return tp / (tp + fn)


def calculate_f1():
    """
            2 * precision * recall / (precision + recall)
    """
    precision = calculate_precision()
    recall = calculate_recall()

    return (2 * precision * recall) / (precision + recall)


if __name__ == "__main__":
    pos = 9
    neg = 5
    entropy = get_entropy(pos, neg)
    pos_we = 6
    pos_st = 3
    neg_we = 2
    neg_st = 3
    info_gain = get_information_gain(positive_examples=pos, positive_weak_examples=pos_we, positive_strong_examples=pos_st,
                                     negative_examples=neg, negative_weak_examples=neg_we, negative_strong_examples=neg_st)
    print(info_gain)
    print()