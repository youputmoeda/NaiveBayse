from numpy import log
import numpy
import pandas as pd
from cf_plot import pretty_plot_confusion_matrix

def data_treatment(emails):

    #Put all lowercase, split and remove non words
    emails['email']=emails['email'].str.replace('\W'," ")
    emails['email']=emails['email'].str.lower()
    emails['email']=emails['email'].str.split()
    #to iterate all emails
    list_of_rows = emails.itertuples(index=False, name='Email')

    aux = list()
    X = list()
    Y = list()
    '''this is to add to Y the labels (ham or spam)
    and its the emails to the list X'''
    for row in list_of_rows:
        Y.append(row[0])
        for z in row[1]:
            if len(z) > 1:
                aux.append(z)
        X.append(aux[:])
        aux.clear()

    return (X, Y)


def data_split(X,Y, train_percentage, test_percentage, validation_percentage):
    '''Split the emails in 70 porcent for training
    15 porcent in validation and
    15 porcent in test'''

    if((train_percentage + test_percentage + validation_percentage) != 1.0):
        print("Error! Data proportions summed are not 1!")
        exit()

    train_end = round(train_percentage*len(Y))
    test_end = round(test_percentage*len(Y)) + train_end

    X_train = X[:train_end]
    Y_train = Y[:train_end]
    X_test = X[train_end:test_end]
    Y_test = Y[train_end:test_end]
    X_validation = X[test_end:]
    Y_validation = Y[test_end:]

    print("Data splited!")
    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation


def train_data(X, Y, C):
    Words_spam = list()
    Words_ham = list()
    vocabulary = list()
    m_spam = 0
    m_ham = 0

    for row in range(len(Y)):
        if Y[row] == 'spam':
            m_spam += 1
            for z in X[row]:
                vocabulary.append(z)
                Words_spam.append(z)
            
        else:
            m_ham += 1
            for z in X[row]:
                vocabulary.append(z)
                Words_ham.append(z)

    #Count each word of spam and ham
    Words_spam_len = {i:Words_spam.count(i) for i in Words_spam}
    Words_ham_len = {i:Words_ham.count(i) for i in Words_ham}
    vocabulary.sort()

    # filtrar e orgaizar os dados
    vocabulary = list(set(vocabulary))

    # Remove words with numbers on it from vocabulary
    words_to_remove = list()
    numbers = '1 2 3 4 5 6 7 8 9'
    for word in vocabulary:
        for letter in word:
            if letter in numbers:
                words_to_remove.append(word)
                break
    for remove in words_to_remove:
        vocabulary.remove(remove)

    #remove common words from vocabulary
    InputFile = open('most_commons_words.txt')
    commons_words = list()
    for line in InputFile:
        commons_words.append(line.strip('\t\n '))
    for word in vocabulary:
        for common in commons_words:
            if word == common:
                vocabulary.remove(word)
    InputFile.close()
    vocabulary.sort()

    #Number of words
    n_words = len(vocabulary)
    #Matriz full of ones with 2 lines and n_words for columns
    R = numpy.ones((2, n_words))

    '''Is to increment the matrix with numbers of times that word appear in spam and ham
    R[0] is the row of spam and R[1] is the row of ham
    and also is to increment the Wham and Wspam that have the value of n_words'''
    index = 0
    Wspam = n_words
    Wham = n_words
    for word in vocabulary:
        if (word in Words_spam_len):
                R[0][index] += Words_spam_len[word]
                Wspam += Words_spam_len[word]
        if (word in Words_ham_len):
                R[1][index] += Words_ham_len[word]
                Wham += Words_ham_len[word]
        index += 1

    #Calculate the probabilities of each word for spam and ham
    for j in range(n_words):
        R[0][j] /= Wspam
        R[1][j] /= Wham

    b = log(C) + log(m_ham) - log(m_spam)
    print("Data Trained!")
    return R, vocabulary, b
   
def test_data(X, Y, R, vocabulary, b):

    '''Testing the Naive Bayse with new emails
    in here we inicialize the variavel t with simetric of b
    we count the number of times each word appears in email and go to the vocabulary getting the position
    in order we know where that word is on the matrix'''
    Y_result = list()
    for row in X:
        t = -b
        Words_len = {i:row.count(i) for i in row}
        for key, value in Words_len.items():
            for i in range(len(vocabulary)):
                if(key == vocabulary[i]):
                    t = t + value*(log(R[0][i]) - log(R[1][i]))

        if(t > 0):
            Y_result.append('spam')
        else:
            Y_result.append('ham')

    print("Data Tested!")

    return Y_result
    



def train_perceptron(X, Y, times_hyperparameter):

    vocabulary = list()

    for row in range(len(Y)):
        for z in X[row]:
            vocabulary.append(z)

    # Remove words with numbers on it from vocabulary
    words_to_remove = list()
    numbers = '1 2 3 4 5 6 7 8 9'
    for word in vocabulary:
        for letter in word:
            if letter in numbers:
                words_to_remove.append(word)
                break
    for remove in words_to_remove:
        vocabulary.remove(remove)

    #remove common words from vocabulary
    InputFile = open('most_commons_words.txt')
    commons_words = list()
    for line in InputFile:
        commons_words.append(line.strip('\t\n '))
    for word in vocabulary:
        for common in commons_words:
            if word == common:
                vocabulary.remove(word)
    InputFile.close()
    vocabulary.sort()

    vocabulary = list(set(vocabulary))
    vocabulary.sort()

    n_words = len(vocabulary)

    x = numpy.zeros((len(Y), n_words + 1))

    line = 0
    for row in range(len(Y)):
        column = 0
        for word in vocabulary:
            for word_in_x in X[row]:
                if (word == word_in_x):
                    x[line][column] += 1
            column += 1
        x[line][column] = 1 # perceptrao nao tem de passar pela origem
        line += 1

    W = numpy.zeros(n_words + 1) # pesos perceptrao

    y = numpy.ones(len(Y))
    for i in range(len(y)):
        if(Y[i] == "spam"):
            y[i] = -1  # spam is represent by -1 and ham by 1
    #times_hyperparameter we choose on the main the value
    for t in range(times_hyperparameter):
        for i in range(len(y)):
            if (y[i]*(W.dot(x[i])) <= 0):
                W = W + y[i]*x[i]

    print("Data for perceptron trained")
    return W, vocabulary


def test_perceptron(X, Y, W, vocabulary):

    n_words = len(vocabulary)

    x = numpy.zeros((len(Y), n_words + 1))

    line = 0
    for row in range(len(Y)):
        column = 0
        for word in vocabulary:
            for word_in_x in X[row]:
                if (word == word_in_x):
                    x[line][column] += 1
            column += 1
        x[line][column] = 1 # perceptrao nao tem de passar pela origem
        line += 1

    Y_result = list()
    for i in range(len(Y)):
        if(W.dot(x[i]) < 0):
            Y_result.append('spam')
        else:
            Y_result.append('ham')

    print("Data for perceptron tested!")

    return Y_result


def results(Y_test, Y_pred):

    accuracy = 0
    for i in range(len(Y_test)):
        if(Y_test[i] == Y_pred[i]):
            accuracy += 1

    accuracy = accuracy/len(Y_test)*100
    error_rate = 1.0 - accuracy/100

    print("Accuracy:")
    print(accuracy)
    print("Error rate:")
    print(error_rate)

    true_spam = 0
    false_spam = 0
    true_ham = 0
    false_ham = 0

    for i in range(len(Y_test)):
        if(Y_test[i] == "spam"):
            if(Y_pred[i] == "spam"):
                true_spam += 1
            else:
                false_ham += 1
        else:
            if(Y_pred[i] == "spam"):
                false_spam += 1
            else:
                true_ham += 1

    confusion_matrix = [[true_spam, false_ham],
                        [false_spam, true_ham]]

    df_cm = pd.DataFrame(confusion_matrix, index = ["Spam", "Ham"], columns = ["Spam", "Ham"])
    pretty_plot_confusion_matrix(df_cm, fz=11, cmap='icefire', figsize=[9,9], show_null_values=2, pred_val_axis='lin')



def main():
    #read spam.csv with columns label and email, it will suffle be himself with the option random_state
    data = pd.read_csv ('spam.csv')
    email = pd.DataFrame(data, columns= ['label', 'email']).sample(random_state=42, frac=1)
    # Treatment of emails
    X, Y = data_treatment(email)

    #Choose perceptrao or Naive Bayse
    print("If you want the perceptron results write P, if you want Naibe Bayse write N")
    while (True):
        helper = input()
        if helper == "P":
            perceptrao = True
            break
        elif helper == "N":
            perceptrao = False
            break
        else:
            print("Wrong code! Try again")
    # proportions
    train = 0.7
    test = 0.15
    validation = 0.15

    X_train, Y_train, X_test, Y_test, X_validation, Y_validation = data_split(X,Y, train, test, validation)

    if(perceptrao):
        # Perceptron
        W, vocabulary = train_perceptron(X_train, Y_train, 5)
        Y_pred = test_perceptron(X_test, Y_test, W, vocabulary)
        test_perceptron(X_validation, Y_validation, W, vocabulary)
    else:    
        # Nayve Bayes
        R , vocabulary, b = train_data(X_train, Y_train, 10)
        Y_pred = test_data(X_test, Y_test, R , vocabulary, b)
        test_data(X_validation, Y_validation, R , vocabulary, b)

    results(Y_test, Y_pred)

    # look for the best hyperparameters
    print("Let's check for the best hyperparameters!")
    accuracy_train = list()
    accuracy_test = list()
    x_axis = list()
    n_iterations = 10
    if (perceptrao):
        for iter in range(n_iterations):
            W, vocabulary = train_perceptron(X_train, Y_train, iter)

            Y_pred = test_perceptron(X_test, Y_test, W, vocabulary)
            accuracy = 0
            for i in range(len(Y_test)):
                if(Y_test[i] == Y_pred[i]):
                    accuracy += 1

            accuracy_test.append(accuracy/len(Y_test))


            Y_pred = test_perceptron(X_train, Y_train, W, vocabulary)
            accuracy = 0
            for i in range(len(Y_train)):
                if(Y_train[i] == Y_pred[i]):
                    accuracy += 1

            accuracy_train.append(accuracy/len(Y_train))

            x_axis.append(iter)
            print("»"*iter, iter,"/", n_iterations - 1)
    else:
        for iter in range(n_iterations):
            R , vocabulary, b = train_data(X_train, Y_train, iter**20)

            Y_pred = test_data(X_test, Y_test, R , vocabulary, b)
            accuracy = 0
            for i in range(len(Y_test)):
                if(Y_test[i] == Y_pred[i]):
                    accuracy += 1

            accuracy_test.append(accuracy/len(Y_test))


            Y_pred = test_data(X_train, Y_train, R , vocabulary, b)
            accuracy = 0
            for i in range(len(Y_train)):
                if(Y_train[i] == Y_pred[i]):
                    accuracy += 1

            accuracy_train.append(accuracy/len(Y_train))

            x_axis.append(iter)
            print("»"*iter, iter,"/", n_iterations - 1)
        
    import matplotlib.pyplot as plt
    plt.plot(x_axis, accuracy_train,'b*', label = 'Train data')
    plt.plot(x_axis, accuracy_test, 'ro', label = 'Test data')
    plt.legend(framealpha=1, frameon=True);
    plt.ylabel('Accuracy')
    plt.xlabel('Number of iterations')
    plt.show()

if __name__ == "__main__":
    main()