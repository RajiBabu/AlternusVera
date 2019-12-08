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

sensationalist_phrases = [

	# From Analyst's Desktop Binder of Homeland Security https://www.scribd.com/doc/82701103/Analyst-Desktop-Binder-REDACTED
	
	'Assassination',
	'Attack',
	'Domestic security',
	'Law enforcement',
	'Disaster',
	'National preparedness',
	'Response',
	'Recovery',
	'Emergency response',
	'First responder',
	'Militia',
	'Shooting',
	'Evacuation',
	'Hostage',
	'Explosion',
	'Organized crime',
	'Gangs',
	'National security',
	'State of emergency',
	'Security breach',
	'Threat',
	'Standoff',
	'Lockdown',
	'Bomb',
	'Riot',
	'Emergency Landing',
	'Incident',
	'Suspicious',
	'Nuclear threat',
	'Hazardous',
	'Infection',
	'Outbreak',
	'Contamination',
	'Terror',
	'Epidemic',
	'Critical Infrastructure',
	'National infrastructure',
	'Transportation security',
	'Grid',
	'Outage',
	'Disruption',
	'Violence',
	'Drug cartel',
	'Narcotics',
	'Shootout',
	'Trafficking',
	'Kidnap',
	'Illegal',
	'Smuggling', 
	'Al Qaeda',
	'Terror attack',
	'Weapon',
	'Improvised explosive device',
	'Suicide bomber',
	'Suicide attack',
	'Hurricane',
	'Tornado',
	'Tsunami',
	'Earthquake',
	'Tremor',
	'Flood',
	'Storm',
	'Extreme weather',
	'Forest fire',
	'Ice',
	'Stranded',
	'Wildfire',
	'Avalanche',
	'Blizzard',
	'Lightening',
	'Emergency Broadcast System',
	'Cyber Security',
	'DDOS',
	'Denial of service',
	'Malware',
	'Phishing',
	'Cyber attack',
	'Cyber terror'
]

def sensphrasedetect(str):
    sum =0
    for x in sensationalist_phrases:
        if x.lower() in str.lower():
            sum+=1
    return sum

def prediction(xtest):
    pickleModel = "/content/gdrive/My Drive/Drifters/Models/Authenticity_Model.pkl"
    pickle_in = open(pickleModel, "rb")
    loadData = pickle.load(pickle_in)
    dataset = loadData.predict(xtest) 
    for i in dataset:
        if(i==0):
            return 0
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
    sensphrCount.append(sensphrasedetect(fnews))
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
        
class Authenticity:
    def __init__(self, fnews):
        self.f_news = processFakeNews(fnews)
        self.xtest = buildSensationalCol(fnews,self.f_news)
        self.x_test = self.xtest 
    def predict(self):
        return prediction(self.x_test)
    def checkNewDataset(self):
        return  newDataset(self.x_test)
