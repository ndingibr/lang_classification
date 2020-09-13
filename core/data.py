
#For Natural language Preprocessing
from nltk.corpus import stopwords
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer

class Feature:
    
    def text_cleaning(self,data):
        #remove duplicates
        data = data.drop_duplicates()
    
        #remove columns with incomplete data
        data = data.dropna(how = 'any')
        
        print(data.head())
        #remove pancuation
        data['text'] = data['text'].str.replace('[^\w\s]','')
        data['language'] = data['language'].str.replace('[^\w\s]','')
    
        #remove stop words. this is a naive approach to remove stop words, based on the english language
        stop = stopwords.words('english')
        data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        data['language'] = data['language'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        
        #convert to lower case
        data['text'] = data['text'].str.lower()
        data['language'] = data['language'].str.lower()
        
        #lemmatization
        data['text'] = data['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        data['language'] = data['language'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
        
        return data   
    
    def feature_engineering(self,data):
        count_vect = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
        count_vect.fit(data['text'])
        return count_vect.transform(data['text'])
    

        
    
