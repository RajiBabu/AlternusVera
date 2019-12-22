import numpy as np
import pandas as pd
import warnings
import pickle
import string
import random
import nltk
nltk.download('punkt')
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

context_Col = [

	# From Analyst's Desktop Binder of Homeland Security https://www.scribd.com/doc/82701103/Analyst-Desktop-Binder-REDACTED
	
	'mailer',
    'news release',
    'press release',
    'CBN News',
    'Fox News',
    'news release',
    'campaign debate',
    'radio ad',
    'news conference',
    'tweet', 
    'radio talk show',
    'campaign ad',
    'TV show season',
    'CPAC conference',
    'chain email',
    'ABC news',
    'CNN',
    'Republican presidential debate',
    'ad',
    'speech',
    'symposium',
    'a rally',
    'NPR',
    'WPRO-AM',
    'committee hearing',
    'Meet the Press',
    'YouTube ad',
    'TV ad',
    'campaign ad',
    'law review article',
    'newspaper article.',
    'Facebook post',
    'vice presidential debate',
    'news story',
    'Face the Nation',
    'robocall',
    'congressional briefing',
    'press release',
    'speech',
    'web pages',
    'Philadelphia',
    'remarks',
    'blog posts',
    'veto message',
    'candidate forum',
    'radio interview',
    'campaign news release',
    'House floor.',
    'Republican National Convention',
    'radio interview.',
    'speech.',
    'campaign material', 
    'email blast',
    'Oregon Senate floor',
    'television ad',
    'Democratic presidential debate',
    'blog post',
    'vice-presidential debate',
    'NBC "Meet the Press"',
    'presidential debate',
    'blog posts',
    'interview on CNN',
    'CNN State of the Union',
    'debate',
    'messages on Twitter',
    'Democratic debate',
    'handout.',
    'The Ed Show',
    'town hall meeting',
    'campaign commercial',
    'Democratic National Convention.',
    'CNN debate ',
    'campaign ad',
    'NBC Commander-In-Chief Forum',
    'South Carolina Democratic presidential debate',
    'post on Facebook',
    'radio show',
    'campaign website',
    'Providence Journal commentary',
    'multiple posts',
    'e-mail',
    'debate.',
    'The Dylan Ratigan Show',
    'remarks to reporters',
    'blog post on RedState.com',
    'email.',
    'television commercial',
    'press release',
    'legislative floor speech',
    'floor speech U.S. Senate',
    'message via Internet'

]

def contextdetect(str):
    sum =0
    for x in context_Col:
        if x.lower() in str.lower():
            sum+=1
    return sum

def prediction(xtest):
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/Authenticity_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)
   # dataset = loadData.predict(xtest) 
   # for i in dataset:
   #     if(i==0):
   #         return 0
    return 1

def processFakeNews(fnews):
    count = lambda l1,l2: sum([1 for x in l1 if x in l2])
    pcCount=[]
    capCount=[]
    digCount=[]
    lenCount=[]
    profanCount=[]
    sensphrCount=[]
    f_news = pd.DataFrame()
    pcCount.append(sum(1 for c in fnews if c=="!" or c=="?"))
    capCount.append(sum(1 for c in fnews if c.isupper()))
    digCount.append(sum(1 for c in fnews if c.isdigit()))
    lenCount.append(len(fnews))
    sensphrCount.append(contextdetect(fnews))
    data = {'puncCount': pcCount, 
        'capCount': capCount,
        'digCount': digCount,
        'lenCount': lenCount,
        'profanCount': profanCount,
        'sensphrCount': sensphrCount}
    f_news['puncCount']=pcCount
    f_news['capCount']=capCount
    f_news['digCount']=digCount
    f_news['lenCount']=lenCount
    f_news['profanCount']=0
    f_news['sensPhrCount']=sensphrCount
    return f_news 

def newDataset(xtest):
    return xtest

def buildSensationalCol(fnews,f_news):
    savedModel = "/content/gdrive/My Drive/Drifters/Models/Authenticity.model"
    contextCol=[]
    model= Doc2Vec.load(savedModel)
    test_data = word_tokenize(fnews.lower())
    v1 = model.infer_vector(test_data)
    similar_doc = model.docvecs.most_similar([v1])
    contextCol.append(similar_doc[0][0])
    contextCol=list(map(int, contextCol))
    f_news['context_new']=contextCol
    return f_news
        
class Authenticity_new:
    def __init__(self, fnews):
        self.f_news = processFakeNews(fnews)
        self.xtest = buildSensationalCol(fnews,self.f_news)
        self.x_test = self.xtest 
    def predict(self):
        return prediction(self.x_test)
    def checkNewDataset(self):
        return  newDataset(self.x_test)
