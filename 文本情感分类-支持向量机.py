# -*- encoding:utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
import re,sklearn
import pandas as pd

df = pd.read_csv('train.csv', lineterminator='\n')
df_test = pd.read_csv('test.csv', lineterminator='\n')

def load_data(type='train'):
    if type=='train':
        data = [review.lower() for review in df['review']]
        data = [(re.sub("[^a-zA-Z]", " ", data[review])) for review in range(len(data))]
        labels = [1 if label=='Positive' else 0 for label in df['label']]
        return data, labels
    else:
        data = [review.lower() for review in df_test['review']]
        data = [(re.sub("[^a-zA-Z]", " ", data[review])) for review in range(len(data))]
        return data


def train_tfidf(train_data):
    tfidf = TFIDF(token_pattern=r"(?u)\b\w+\b") # 0.85030136
    tfidf.fit(train_data)
    return tfidf


def train_SVC(data_vec, label):
    SVC = sklearn.svm.SVC(kernel='linear',probability=True)
    #clf = CalibratedClassifierCV(SVC)
    SVC.fit(data_vec, label)
    return SVC
def train():
    train_data, labels = load_data('train')
    tfidf = train_tfidf(train_data)
    train_vec = tfidf.transform(train_data)
    model = train_SVC(train_vec, labels)
    print('model saving...')
    joblib.dump(tfidf, 'model/tfidf.model')
    joblib.dump(model, 'model/svc.model')
def predict():
    test_data = load_data('test')
    print('load model...')
    tfidf = joblib.load('model/tfidf.model')
    model = joblib.load('model/svc.model')
    test_vec = tfidf.transform(test_data)
    test_predict = model.predict_proba(test_vec)
    return test_predict

def train_no_save_model():
    train_data, labels = load_data('train')
    tfidf = train_tfidf(train_data)
    train_vec = tfidf.transform(train_data)
    model = train_SVC(train_vec, labels)
    test_data = load_data('test')
    test_vec = tfidf.transform(test_data)
    test_predict = model.predict_proba(test_vec)
    return test_predict

# train()
# test_predict = predict() # 保存模型预测
test_predict = train_no_save_model()
test_predict_positive = [item[1] for item in test_predict]
print(test_predict[:5])

# 写入预测文件，提交结果
test_ids = df_test['ID']
Data = {'ID':test_ids, 'Pred':test_predict_positive}
pd.DataFrame(Data, columns=['ID', 'Pred']).to_csv('test_pred.csv', header=True) #写入文件
print('Done')