from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

def load_data(): # loading and processing data
    
    labels, data = [], []
    
    with open("clean_real.txt", 'r') as real: # reading "real" news headlines
        pullCleanData = real.readlines()
        for line in pullCleanData:
            data.append(line[: -1])
            labels.append(1) # every real news headline is labelled by 1
            
    with open("clean_fake.txt", 'r') as fake:
        pullFakeData = fake.readlines()
        for line in pullFakeData:
            data.append(line[: -1])
            labels.append(0) # every fake news headline is labelled by 0
            
    # our goal is to classify news as 1 or 0, based on the number of occurences of certain words
    # hence we make use of CountVectorizer()
    coun_vect = CountVectorizer()
    count_matrix = coun_vect.fit_transform(data)
    finalData = count_matrix.toarray()
    
    # 70% of the data goes to the training set, 15% to the validation set, and 15% to the test set
    x_train, x_test, y_train, y_test = train_test_split(finalData, labels, test_size = 0.3, random_state = 42)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size = 0.5, random_state = 42)

    # return the three sets with their respective lables
    return x_train, x_test, x_val, y_train, y_test, y_val 
    
def select_knn_model():

    # get the training, validation, and test sets with their corresponding lables
    x_train, x_test, x_val, y_train, y_test, y_val = load_data()

    # keep track of the model with the best accuracy on the validation set
    kWithBestValidationAccuracy,  bestValidationAccuracy= -1, -1  

    # we train 20 different models as k goes from 1 to 20; we find the most accurate model
    for k in range(1, 21):
        knn = KNeighborsClassifier(n_neighbors = k, metric = 'cosine')

        # train the model on the training set
        knn.fit(x_train, y_train)

        # compute the accuracy of the model on the training set
        y_predOnTrain = knn.predict(x_train)
        accuracyOnTrain = metrics.accuracy_score(y_train, y_predOnTrain)

        # compute the accuracy of the model on the validation set
        y_predOnVal = knn.predict(x_val)
        accuracyOnVal = metrics.accuracy_score(y_val, y_predOnVal)

        # check if the current model is the best so far, and update the variables if necessary
        if accuracyOnVal > bestValidationAccuracy:
            kWithBestValidationAccuracy = k
            bestValidationAccuracy = accuracyOnVal
            
        print("For k =", k, ", the training error is ", 1 - accuracyOnTrain)
        print("For k =", k, ", the validation error is ", 1 - accuracyOnVal, '\n')

        # plot the two accuracies vs k
        plt.scatter(k, accuracyOnTrain, color = 'green')
        plt.scatter(k, accuracyOnVal, color = 'blue')

    print('Clearly, the best accuracy on the validation set corresponds to k = ', kWithBestValidationAccuracy, '\n')

    # these points are not plotted; they are here to just add the labels
    plt.plot(1, 0.5, color = 'green', label = 'Accuracy On Training set')
    plt.plot(1, 0.5, color = 'blue', label = 'Accuracy On Validation set')
    
    # report the accuracy of the best model on the test set
    knn = KNeighborsClassifier(n_neighbors = kWithBestValidationAccuracy, metric = 'cosine')
    knn.fit(x_train, y_train)
    y_predOnTest = knn.predict(x_test)
    bestAccuracyOnTest = metrics.accuracy_score(y_test, y_predOnTest)
    
    print('Accuracy on the test set corresponding to the model with best validation accuracy (i.e. when k =', 
          kWithBestValidationAccuracy, ') = ', bestAccuracyOnTest)

    # customize the plot
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs k")
    plt.legend()
    plt.show()

# call the function
select_knn_model()
    
