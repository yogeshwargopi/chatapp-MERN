#general purpose packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#data processing
import re, string
import emoji
import nltk

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split


#Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

#transformers
from transformers import BertTokenizerFast
from transformers import TFBertModel
from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel

#keras
import tensorflow as tf
from tensorflow import keras


#metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('C:\webpage\project\chatapp\server\Mlmodel\ead.csv',encoding='ISO-8859-1')


sentence_urgency = df['label'].value_counts().loc[lambda x : x > 100].reset_index(name='counts')

#preprocessing
# defininig a fuction which takes a column of sentences and does preprocessing of texts of sentences
nltk.download('stopwords')
nltk.download('wordnet')
def cleaning(df1):
    lowered=df1.lower()    # lowering the sentences
    removed=re.sub(r'[^a-z]',' ',lowered)  # replacing the non alphabets with space
    splitted=removed.split(' ')   # splitting the sentences by spaces
    df1= [WordNetLemmatizer().lemmatize(word) for word in splitted if word not in stopwords.words('english')]  # lemmatizing and removing stopwords from list
    df1=' '.join(df1) # joining back the words of list
    return(df1) # returning the cleaned words

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
df['sentence']=df['sentence'].apply(cleaning)


df['label'].value_counts()

df['label'] = df['label'].map({'urgent':0,'non-urgent':1})

df['label'].value_counts()

X = df['sentence'].values
y = df['label'].values

seed=42

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, stratify=y, random_state=seed)

y_train_le = y_train.copy()
y_valid_le = y_valid.copy()

ohe = preprocessing.OneHotEncoder()
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
y_valid = ohe.fit_transform(np.array(y_valid).reshape(-1, 1)).toarray()



clf = CountVectorizer()
X_train_cv =  clf.fit_transform(X_train)
X_test_cv = clf.transform(X_valid)

tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_cv)
X_train_tf = tf_transformer.transform(X_train_cv)
X_test_tf = tf_transformer.transform(X_test_cv)

MAX_LEN=128

def tokenize(data,max_len=MAX_LEN) :
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return np.array(input_ids),np.array(attention_masks)


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

train_input_ids, train_attention_masks = tokenize(X_train, MAX_LEN)
val_input_ids, val_attention_masks = tokenize(X_valid, MAX_LEN)
#test_input_ids, test_attention_masks = tokenize(X_test, MAX_LEN)

bert_model = TFBertModel.from_pretrained('bert-base-uncased')

def create_model(bert_model, max_len=MAX_LEN):

    ##params###
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=1e-7)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()


    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')

    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')

    embeddings = bert_model([input_ids,attention_masks])[1]

    output = tf.keras.layers.Dense(2, activation="softmax")(embeddings)

    model = tf.keras.models.Model(inputs = [input_ids,attention_masks], outputs = output)

    model.compile(opt, loss=loss, metrics=accuracy)


    return model

model = create_model(bert_model, MAX_LEN)

history_bert = model.fit([train_input_ids,train_attention_masks], y_train, validation_data=([val_input_ids,val_attention_masks], y_valid), epochs=4, batch_size=32)

result_bert = model.predict([val_input_ids,val_attention_masks])

y_pred_bert =  np.zeros_like(result_bert)
y_pred_bert[np.arange(len(y_pred_bert)), result_bert.argmax(1)] = 1

from sklearn.metrics import confusion_matrix as conf_matrix,ConfusionMatrixDisplay, classification_report


cm=conf_matrix(y_valid.argmax(1), y_pred_bert.argmax(1))

print('\tClassification Report for BERT:\n\n',classification_report(y_valid,y_pred_bert, target_names=['urgent', 'non-urgent']))


#val_input_ids, val_attention_masks = tokenize(X_valid, MAX_LEN)
testsentence=[]
testsentence.append('join the meeting its urgent')
testsentence.append('join')

mytest_input_ids, mytest_attention_masks = tokenize(testsentence, MAX_LEN)

myresult_bert = model.predict([mytest_input_ids,mytest_attention_masks])

threshold = 0.5
predictions = (myresult_bert[:, 0] > threshold).astype(int)
predictions = [bool(pred) for pred in predictions]

print(predictions)