from numpy import log
import numpy
import pandas as pd


def train(emails, emails_classification):
    row1 = emails.itertuples(index=False, name='Email')
    Words = list()
    emails['v2']=emails['v2'].str.replace('\W'," ")
    emails['v2']=emails['v2'].str.lower()
    emails['v2']=emails['v2'].str.split()
    for i in emails['v2']:
        for z in i:
            Words.append(z)
    Words = list(set(Words))
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
    Wham = N_words
    Wspam = N_words
    j = 1
    while (j != range(N_words)):
        for h in spam_word:
            R[0][j] = h
            print("matriz:",R[0][j])
            print (j)
            Wspam += h
            j += 1
    print(R)
    return
    row = spam_messages.itertuples()
    for h in row:
        print(h)
        print(h[50])
    return
    s = 0
    while (s != len(aux)):
        x = s + 1
        Word_email = 1
        while (x != len(aux)):
            if aux[x] == aux[s]:
                Word_email += 1
                aux[x] = ''
            x += 1
        if aux[s] == '':
            s += 1
            continue
        s += 1
        j += 1
    else:
        helperrr = z[1]
        s = 0
        while (s != len(helperrr)):
            x = s + 1
            Word_email = 1
            while (x != len(helperrr)):
                if helperrr[x] == helperrr[s]:
                    Word_email += 1
                    helperrr[x] = ''
                x += 1
            if helperrr[s] == '':
                s += 1
                continue
            s += 1
            R[1][j_Second] = Word_email
            Wham += Word_email
            j_Second += 1
    i += 1
    return
    print("Wham:",Wham)
    print("Wspam:",Wspam)
    probability(N_words, R, Wham, Wspam)
    classification(R, b, N_words, C, emails_classification)

def probability(N_words, R, Wham, Wspam):
    s = 0
    while (s != N_words):
        R[0][s] = R[0][s]/Wham
        R[1][s] = R[1][s]/Wspam
        s += 1
    return (R)
    
def classification(R, b, N_words, C, emails_classification):
    h = 0
    row = emails_classification.itertuples(index=False, name='Email')
    lista = list()
    for z in row:
        aux = z[1].split()
        s = 0
        while (s != len(aux)):
            x = s + 1
            Word_teste = 1
            while (x != len(aux)):
                if aux[x] == aux[s]:
                    Word_teste += 1
                    aux[x] = ''
                x += 1
            if aux[s] == '':
                s += 1
                continue
            lista.append(Word_teste)
            s += 1
        t = -b
        print("t", t)
        while (h != len(lista)):
            for s in range(N_words):
                t = t + lista[h]*(log(R[0][s]) - log(R[1][s]))
            h += 1
        k = 0
        while (k != len(C)):
            print("k", k)
            print(t[k])
            if t[k] > 0:
                print("spam")
                print (z)
            else:
                print("Ham")
                print (z)
            k += 1

def main():
    data = pd.read_csv (r'spam.csv')
    emails = pd.DataFrame(data, columns= ['v1', 'v2']).sample(random_state=1,frac=0.7) #Treino só se usa 70 porcento
    emails = pd.DataFrame(data, columns= ['v1', 'v2']).sample(random_state=1,frac=0.7) #Treino só se usa 70 porcento

    emails_probability = pd.DataFrame(data, columns= ['v1', 'v2']).sample(frac=0.15)
    emails_classification = pd.DataFrame(data, columns= ['v1', 'v2']).sample(frac=0.15)
    train(emails, emails_classification)
    

if __name__ == "__main__":
    main()