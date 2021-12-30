import re
import pandas as pd 

sms_data= pd.read_csv('spam.csv')
print(sms_data.shape)
print(sms_data.head())
print(sms_data['v1'].value_counts(normalize='True')*100)
data_sample=sms_data.sample(frac=1,random_state=1)
random_data=round(len(data_sample)*0.7)
train_set=data_sample[:random_data].reset_index(drop=True)
test_set=data_sample[random_data:].reset_index(drop=True)
print(train_set)
print(test_set)
print(train_set['v1'].value_counts(normalize=True))
print(test_set['v1'].value_counts(normalize=True))
train_set['v2']=train_set['v2'].str.replace('\W'," ")
train_set['v2']=train_set['v2'].str.lower()
train_set.head()
train_set['v2']=train_set['v2'].str.split()
vocabulary = []
for v2 in train_set['v2']:
    for word in v2:
        vocabulary.append(word)
        
vocabulary = list(set(vocabulary))
word_counts_per_v2 = {unique_word: [0] * len(train_set['v2']) for unique_word in vocabulary}

for index, v2 in enumerate(train_set['v2']):
    for word in v2:
        word_counts_per_v2[word][index] += 1

word_col=pd.DataFrame(word_counts_per_v2)
word_col.head()
training_set=pd.concat([train_set,word_col],axis=1)
training_set.head()
prob=training_set['v1'].value_counts(normalize=True)
p_ham,p_spam=prob[0],prob[1]
spam_messages = training_set[training_set['v1'] == 'spam']
ham_messages = training_set[training_set['v1'] == 'ham']
spam_word=spam_messages['v2'].apply(len)
n_spam=spam_word.sum()

ham_word=ham_messages['v2'].apply(len)
n_ham=ham_word.sum()

n_vocabulary=len(vocabulary)
alpha=1
print(n_vocabulary)
param_spam={word:0 for word in vocabulary}
param_ham={word:0 for word in vocabulary}
for word in vocabulary:
    spam_w=spam_messages[word].sum()
    p_w_given_spam=(spam_w+alpha)/(n_spam+alpha*n_vocabulary)
    param_spam[word]=p_w_given_spam
   
    ham_w=ham_messages[word].sum()
    p_w_given_ham=(ham_w+alpha)/(n_ham+alpha*n_vocabulary)
    param_ham[word]=p_w_given_ham


def classify(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message=p_spam
    p_ham_given_message=p_ham
    for word in message:
        if word in param_spam:
            p_spam_given_message*=param_spam[word]
        if word in param_ham:
            p_ham_given_message*=param_ham[word]
        
    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')

classify('WINNER!! This is the secret code to unlock the money: C3421.')
classify("Sounds good, Tom, then see u there")

def classify_test_set(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in message:
        if word in param_spam:
            p_spam_given_message *= param_spam[word]

        if word in param_ham:
            p_ham_given_message *= param_ham[word]

    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_spam_given_message > p_ham_given_message:
        return 'spam'
    else:
        return 'needs human classification'

test_set['predicted'] = test_set['v2'].apply(classify_test_set)
print(test_set.head())
total_test_case=test_set.shape[0]
correct=0
for row in test_set.iterrows():
    row = row[1]
    if row['v1']==row['predicted']:
        correct+=1
        
print("correct : ",correct)
print("Total messages:",total_test_case)
print("Accuracy:",correct/total_test_case)