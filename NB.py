import seaborn as sns
import matplotlib.pyplot as plt
from numpy import log
import numpy
import pandas as pd

def train_data(emails):
    Words_spam = list()
    Words_ham = list()
    vocabulary = list()
    m_spam = 0
    m_ham = 0
    emails['v2']=emails['v2'].str.replace('\W'," ")
    emails['v2']=emails['v2'].str.lower()
    emails['v2']=emails['v2'].str.split()
    row = emails.itertuples(index=False, name='Email')
    for i in row:
        if i[0] == 'spam':
            m_spam += 1
            for z in i[1]:
                if len(z) > 1:
                    vocabulary.append(z)
                    Words_spam.append(z)
            print()
            Words_spam_len = {i:Words_spam.count(i) for i in Words_spam}
            
        else:
            m_ham += 1
            for z in i[1]:
                if len(z) > 1:
                    vocabulary.append(z)
                    Words_ham.append(z)
            Words_ham_len = {i:Words_ham.count(i) for i in Words_ham}

    print(Words_ham_len)
    for i in Words_spam_len:
        print(Words_spam_len[i])
    n_words = len(vocabulary)
    print(n_words)
    m = m_spam + m_ham
    
    Words_spam = list(set(Words_spam))
    Words_ham = list(set(Words_ham))

    C = 1, 5, 10
    b = log(C) + log(m_ham) - log(m_spam)

    R = numpy.ones((2, n_words))

    row = emails.itertuples(index=False, name='Email')
    for i in range(m):
        for k in row:
            if k[0] == 'spam':
                for l in Words_spam_len:
                    for j in range(n_words):
                        print(Words_spam_len[l])
                        R[0][j] += Words_spam_len[l]
                        print(R[0][j])
                    print("something")
            else:
                for j in range(n_words):
                    R[1][j] += Words_ham_len[l]

    Wspam = len(Words_spam)
    Wham = len(Words_ham)
    for j in range(n_words):
        R[0][j] /= Wspam
        R[1][j] /= Wham
    return
    word_counts_per_v2 = {unique_word: [0] * len(emails['v2']) for unique_word in Words} #Criar palavras unicas
    for index, v2 in enumerate(emails['v2']):
        for word in v2:
            word_counts_per_v2[word][index] += 1
    word_col=pd.DataFrame(word_counts_per_v2)
    training_set=pd.concat([emails,word_col],axis=1)
    prob=training_set['v1'].value_counts(normalize=True) #If True then the object returned will contain the relative frequencies of the unique values.
    p_ham,p_spam = prob[0],prob[1]


    #Estas duas linhas servem para separar em duas tabelas, uma com o spam e a outro com o ham
    spam_messages = training_set[training_set['v1'] == 'spam']
    ham_messages = training_set[training_set['v1'] == 'ham']
    
    #conta quantas vezes aparece cada palavra no spam e no ham respetivamente
    spam_word = spam_messages['v2'].apply(len) 
    ham_word = ham_messages['v2'].apply(len)
    
    #Numero total de palavras no spam, no ham e no total respetivamente
    n_spam = spam_word.sum() 
    n_ham = ham_word.sum() 
    N_words = len(Words)

    #contar numero de emails de spam, ham e total
    m_spam = len(spam_messages['v1'])
    m_ham = len(ham_messages['v1'])
    m_email = m_spam + m_ham

    C = 1, 5, 10
    b = log(C) + log(m_ham) - log(m_spam)
    print("Nº of mails",m_email)
    print("Spam:",m_spam)
    print("Ham:",m_ham)
    print('---------------')
    print ("C =",C)
    print("log(ham):", log(m_ham))
    print("log(spam):", log(m_spam))
    print("B(C = 1):",b[0], "B(C = 5):",b[1],"B(C = 10):",b[2])
    print('---------------')
    R = numpy.ones((2, N_words))
    print(R)
    Wham = n_ham
    Wspam = n_spam
    j = 0
    print(spam_word)
    for h in spam_word:
        return
        R[0][j] = h
        j += 1
    print(R)
    j = 0
    for h in ham_word:
        R[1][j] = h
        j += 1
    print(R)
    print("Wham:",Wham)
    print("Wspam:",Wspam)
    # print(emails_classification)
    # probability(N_words, R, Wham, Wspam)
    # classification(R, b, N_words, C, emails_classification)

def probability(N_words, R, Wham, Wspam):
    for j in range(N_words):
        R[0][j] = R[0][j]/Wham
        R[1][j] = R[1][j]/Wspam
    return (R)
    
def classification(R, b, N_words, C, emails_classification):
    h = 0
    t = -b
    lista_words = list()
    for email in emails_classification['v2']:
        email = email.split()
        for word in email:
            word_len = email.count(word)
            lista_words.append(word_len)
            while (h != len(lista_words)):
                for s in range(N_words):
                    t *= lista_words[h]*(log(R[0][s]) - log(R[1][s]))
                h += 1

    k = 0
    while (k != len(C)):
        print("k", k)
        print(t[k])
        if t[k] > 0:
            print("spam")
        else:
            print("Ham")
        k += 1

def train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=None):
    '''
    Esta função serve para separar a Dataframe(os emails) em 3 partes
    o treino que é 70 porcento
    o teste que é 15 porcento 
    e a validação que é 15 porcento
    '''
    numpy.random.seed(seed)
    perm = numpy.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.iloc[perm[:train_end]]
    validate = df.iloc[perm[train_end:validate_end]]
    test = df.iloc[perm[validate_end:]]
    return train, validate, test

def main():
    # data = pd.read_csv (r'spam.csv')
    data = pd.read_csv (r'try.csv')
    email = pd.DataFrame(data, columns= ['v1', 'v2']).sample(random_state=42, frac=1)

    # train, validate, test = train_validate_test_split(data_sample)

    # print(train)
    # print(len(validate))
    # print(len(test))



    train_data(email)
    

if __name__ == "__main__":
    main()