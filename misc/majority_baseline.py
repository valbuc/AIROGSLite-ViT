import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, accuracy_score


def majority_baseline(training_labels, testing_labels):
    # determining the majority class based on the training data
    train_label_freq = Counter()
    for img_label in training_labels:
        train_label_freq.update(str(img_label))
    majority_class = int(train_label_freq.most_common(1)[0][0])

    # assign majority label to all test instances as prediction
    prediction_list = [majority_class for _ in testing_labels]

    return prediction_list


def create_label_list():
    # read image info
    img_info_df = pd.read_csv("img_info.csv")
    # sort dataframe based on shuffled name
    img_info_df = img_info_df.sort_values("new_file", ignore_index=True)
    # create list with labels of training data
    training_data_labels = img_info_df.iloc[0:10500]["labels_int"].tolist()
    # create list with labels of test data
    test_data_labels = img_info_df.iloc[13500:15000]["labels_int"].tolist()

    return training_data_labels, test_data_labels


if __name__ == '__main__':
    # create lists with labels of training and test data
    train_labels, test_labels = create_label_list()

    # make predictions based on majority baseline
    predictions = majority_baseline(train_labels, test_labels)

    # calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)
    # calculate f1 score
    f1 = f1_score(test_labels, predictions, pos_label=1, average='binary', zero_division='warn')

    print(accuracy)
    print(f1)
