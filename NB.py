from numpy import log
import numpy
import pandas as pd


def train(emails, emails_classification):
    ham = 0
    spam = 0
    row = emails.itertuples(index=False, name='Email')
    Bag = ['an', 'a', 'on', 'in', 'at']
    Words = list()
    MAILS = list()
    for i in emails['v2']:
        j = i.split()
        MAILS.append(j)
        for z in j:
            if len(z) != 0: #remove empty words
                if len(z) != 1: #remove singles word a, b, c
                    if (len(z) != 2): #remove word with size 2, because they all are super common, an, it, at, we, us
                        # z = z.strip(" .,0123456789-\\?+-)(�"";#&''=*")
                        # z.split(".")
                        Words.append(z)
    N = len(Words)
    print("lenght:", N)
    # Sample = emails.sample(axis=0) #Gonna use this later
    print()
    while (emails.empty == False):
        for aux in emails.itertuples(index=False, name='Email'):
            if (aux[0] == 'ham'):
                ham = ham + 1
            else:
                spam = spam + 1
            emails = emails.iloc[1: ,]
    m = spam + ham
    print("Nº of mails",m)
    print("Spam:",spam)
    print("Ham:",ham)
    print('---------------')
    C = 1, 5, 10
    print ("C =",C)
    print("log(ham):", log(ham))
    print("log(spam):", log(spam))
    b = log(C) + log(ham) - log(spam)
    print("B(C = 1):",b[0], "B(C = 5):",b[1],"B(C = 10):",b[2])
    print('---------------')
    R = numpy.ones((2, N))
    Wham = N
    Wspam = N
    i = 0
    j_Second = 0
    j = 0
    while (i != m):
        for z in row:
            if z[0] == 'spam':
                aux = z[1].split()
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
                    R[0][j] = Word_email
                    Wspam += Word_email
                    j += 1
            else:
                helperrr = z[1].split()
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
    print("Wham:",Wham)
    print("Wspam:",Wspam)
    probability(N, R, Wham, Wspam)
    classification(R, b, N, C, emails_classification)

def probability(N, R, Wham, Wspam):
    s = 0
    while (s != N):
        R[0][s] = R[0][s]/Wham
        R[1][s] = R[1][s]/Wspam
        s += 1
    return (R)
    
def classification(R, b, N, C, emails_classification):
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
            for s in range(N):
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
    emails = pd.DataFrame(data, columns= ['v1', 'v2']).sample(frac=0.7) #Treino só se usa 70 porcento
    emails_probability = pd.DataFrame(data, columns= ['v1', 'v2']).sample(frac=0.15)
    emails_classification = pd.DataFrame(data, columns= ['v1', 'v2']).sample(frac=0.15)
    train(emails, emails_classification)
    

if __name__ == "__main__":
    main()