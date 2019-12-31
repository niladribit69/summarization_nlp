from urllib import request                                            #importing libraries
from bs4 import BeautifulSoup as bs
import re
import nltk

url="https://en.wikipedia.org/wiki/Machine_learning"                  #loading datas from wikipedia
htmlDoc=request.urlopen(url)
soupObject=bs(htmlDoc,'html.parser')
paragraphContents=soupObject.findAll('p')
print(paragraphContents)

allParagraphContent=""                                               #takes out the paragraph from the html page of wikipedia
for paragraphcontent in paragraphContents:
    allParagraphContent+=paragraphcontent.text
    print(allParagraphContent)
    
allParagraphData_cleanedData=re.sub(r'\[[0-9]*\]',' ',allParagraphContent)                         #cleaning the data
allParagraphData_cleanedData=re.sub(r'\s+',' ',allParagraphData_cleanedData)
sentences_tokens=nltk.sent_tokenize(allParagraphData_cleanedData)
allParagraphData_cleanedData=re.sub(r'[^a-zA-Z]',' ',allParagraphData_cleanedData)
allParagraphData_cleanedData=re.sub(r'\s+',' ',allParagraphData_cleanedData)
print(allParagraphData_cleanedData) 

words_tokens=nltk.word_tokenize(allParagraphData_cleanedData)        #creating the word tokens
print(words_tokens)

from nltk.corpus import stopwords                                    #removing stop-words and counting the word frequency
stopwords=stopwords.words("english")
word_frequencies={}
for word in words_tokens:
    if word not in stopwords:
        if word not in word_frequencies.keys():
            word_frequencies[word]=1
        else:
            word_frequencies[word]+=1
print(word_frequencies)        

max_frequency_word=max(word_frequencies.values())                   #calculate weighted frequency
for word in word_frequencies.keys():
    word_frequencies[word]=(word_frequencies[word]/max_frequency_word)
print(word_frequencies)

sentence_scores={}

for sentence in sentences_tokens:                                  #create sentence score with each weighted frequency
    for word in nltk.word_tokenize(sentence.lower()):
        if word in word_frequencies.keys():
            if (len(sentence.split(' ')))<30:
                if sentence not in sentence_scores.keys():
                    sentence_scores[sentence]=word_frequencies[word]
                else:
                    sentence_scores[sentence]+=word_frequencies[word]

print(sentence_scores)  

import heapq                                                     #importing heapq and generating summary
summary=heapq.nlargest(5,sentence_scores,key=sentence_scores.get)
print(summary)
            
