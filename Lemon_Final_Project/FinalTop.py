# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:51:52 2019

@author: Lemon Lin Reimer
"""

#%% packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from MLPackage.FinalML import Jesse
import gensim
import gensim.corpora as corpora
from FinalCrawler import mycrawler
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#%% import data
data = pd.read_csv("C:/Users/swagj/Documents/NLPforSS/Lemon_Final_Project/newporn.csv")

data = data.dropna(subset = ['cleancomment'])

classifier = list()

for val in data.search:
    if val == "POV" or val == "VR":
        classifier.append("First")
    else:
        classifier.append("Third")
        
data['classifier'] = classifier

train, test = train_test_split(data, random_state = 1234)

#%% prep for theme extraction
themes = list()

for word in data.cleancomment.str.lower():
    themes.append(word.split(" "))

data['themes'] = themes

#%% theme extraction with stops
First = corpora.Dictionary(data.themes[data.classifier == 'First'])
corpus = [First.doc2bow(text) for text in data.themes[data.classifier == 'First']]

ldamod = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word = First, passes = 45)
First_topics = ldamod.print_topics(num_words = 10)

print("Comment Topics for Participatory Videos")
for topic in First_topics:
    print(topic)

Third = corpora.Dictionary(data.themes[data.classifier == 'Third'])
corpus2 = [Third.doc2bow(text) for text in data.themes[data.classifier == 'Third']]

ldamod2 = gensim.models.ldamodel.LdaModel(corpus2, num_topics = 5, id2word = Third, passes = 45)
Third_topics = ldamod2.print_topics(num_words = 10)

print("Comment Topics for Observational Videos")
for topic in Third_topics:
    print(topic)

#%% Prepping the lemma
    
crawl = mycrawler()
stops = stopwords.words('english')
lemmer = WordNetLemmatizer()
lemma = []

for com in data.cleancomment.str.lower():
    tokenwrd = word_tokenize(com)
    tmp2 = [lemmer.lemmatize(wrd, crawl.get_pos(wrd)) for wrd in tokenwrd if wrd not in stops]
    lemma.append(tmp2)

data['lemma'] = lemma

crawl.teardown()

#%% theme extraction without stops and lemma'd
First = corpora.Dictionary(data.lemma[data.classifier == 'First'])
corpus = [First.doc2bow(text) for text in data.lemma[data.classifier == 'First']]

ldamod = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word = First, passes = 45)
First_topics = ldamod.print_topics(num_words = 10)

print("Comment Topics for Participatory Videos")
for topic in First_topics:
    print(topic)

Third = corpora.Dictionary(data.lemma[data.classifier == 'Third'])
corpus2 = [Third.doc2bow(text) for text in data.lemma[data.classifier == 'Third']]

ldamod2 = gensim.models.ldamodel.LdaModel(corpus2, num_topics = 5, id2word = Third, passes = 45)
Third_topics = ldamod2.print_topics(num_words = 10)

print("Comment Topics for Observational Videos")
for topic in Third_topics:
    print(topic)
    
#%% Overall topic model (lemma w/o stops)
    
Total = corpora.Dictionary(data.lemma)
corpus = [First.doc2bow(text) for text in data.lemma]

ldamod = gensim.models.ldamodel.LdaModel(corpus, num_topics = 5, id2word = Total, passes = 45)
First_topics = ldamod.print_topics(num_words = 10)

print("Comment Topics in General")
for topic in First_topics:
    print(topic)

#%% Vectorization
mytfidf = TfidfVectorizer(ngram_range=(1,2))

df_tfidf = mytfidf.fit_transform(train.cleancomment).toarray()
colnames = mytfidf.get_feature_names()
df_tfidf = pd.DataFrame(df_tfidf, columns = colnames)

mycount = CountVectorizer(ngram_range=(1,2))

df_count = mycount.fit_transform(train.cleancomment).toarray()
colnames2 = mycount.get_feature_names()
df_count = pd.DataFrame(df_count, columns = colnames2)

#%% count Model
ML = Jesse()

my_dim2, pca2 = ML.iterate_var(df_count, 0.95, 200)

param_grid2 = {"max_depth": [10, 50, 100],
              "n_estimators": [16, 32, 64],
              "random_state": [1234]}

clf_pca2 = RandomForestClassifier()

gridsearch_model2, best2, opt_params2 = ML.grid_search(
        param_grid2, clf_pca2, df_count, train.search)

clf_pca2 = RandomForestClassifier()
clf_pca2.set_params(**gridsearch_model2.best_params_)
clf_pca2.fit(my_dim2, train.search)

#%% tfidf Model
ML = Jesse()

my_dim, pca = ML.iterate_var(df_tfidf, 0.95, 200)

param_grid = {"max_depth": [10, 50, 100],
              "n_estimators": [16, 32, 64],
              "random_state": [1234]}

clf_pca = RandomForestClassifier()

gridsearch_model, best, opt_params = ML.grid_search(
        param_grid, clf_pca, df_tfidf, train.search)

clf_pca = RandomForestClassifier()
clf_pca.set_params(**gridsearch_model.best_params_)
clf_pca.fit(my_dim, train.search)

#%% test
tfidf_predictions = ML.prediction(test.cleancomment, test.search, mytfidf, pca, clf_pca)

count_predictions = ML.prediction(test.cleancomment, test.search, mycount, pca2, clf_pca2)

#%% metrics
accuracy = float(list(
        (tfidf_predictions.actual == tfidf_predictions.predicted)
        ).count(True))/float(len(tfidf_predictions))
the_measures = pd.DataFrame(
        precision_recall_fscore_support(tfidf_predictions.actual,
                                        tfidf_predictions.predicted)).T
the_measures.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures.index = ['Cuckold', 'Hidden_Cam', 'POV', 'VR']
the_measures.accuracy = accuracy
print("tfidf measures")
print (the_measures)

accuracy2 = float(list(
        (count_predictions.actual == count_predictions.predicted)
        ).count(True))/float(len(count_predictions))
the_measures2 = pd.DataFrame(
        precision_recall_fscore_support(count_predictions.actual,
                                        count_predictions.predicted)).T
the_measures2.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures2.index = ['Cuckold', 'Hidden_Cam', 'POV', 'VR']
the_measures2.accuracy = accuracy2
print("count measures")
print (the_measures2)

#%% new predicted variable: camera angle

#%% new Vectorization
mytfidf = TfidfVectorizer(ngram_range=(1,2))

df_tfidf = mytfidf.fit_transform(train.cleancomment).toarray()
colnames = mytfidf.get_feature_names()
df_tfidf = pd.DataFrame(df_tfidf, columns = colnames)

mycount = CountVectorizer(ngram_range=(1,2))

df_count = mycount.fit_transform(train.cleancomment).toarray()
colnames2 = mycount.get_feature_names()
df_count = pd.DataFrame(df_count, columns = colnames2)

#%% new count Model
ML = Jesse()

my_dim2, pca2 = ML.iterate_var(df_count, 0.95, 200)

param_grid2 = {"max_depth": [10, 50, 100],
              "n_estimators": [16, 32, 64],
              "random_state": [1234]}

clf_pca2 = RandomForestClassifier()

gridsearch_model2, best2, opt_params2 = ML.grid_search(
        param_grid2, clf_pca2, df_count, train.classifier)

clf_pca2 = RandomForestClassifier()
clf_pca2.set_params(**gridsearch_model2.best_params_)
clf_pca2.fit(my_dim2, train.classifier)

#%% new tfidf Model
ML = Jesse()

my_dim, pca = ML.iterate_var(df_tfidf, 0.95, 200)

param_grid = {"max_depth": [10, 50, 100],
              "n_estimators": [16, 32, 64],
              "random_state": [1234]}

clf_pca = RandomForestClassifier()

gridsearch_model, best, opt_params = ML.grid_search(
        param_grid, clf_pca, df_tfidf, train.classifier)

clf_pca = RandomForestClassifier()
clf_pca.set_params(**gridsearch_model.best_params_)
clf_pca.fit(my_dim, train.classifier)

#%% new test
tfidf_predictions = ML.prediction(test.cleancomment, test.classifier, mytfidf, pca, clf_pca)

count_predictions = ML.prediction(test.cleancomment, test.classifier, mycount, pca2, clf_pca2)

#%% new metrics
accuracy = float(list(
        (tfidf_predictions.actual == tfidf_predictions.predicted)
        ).count(True))/float(len(tfidf_predictions))
the_measures = pd.DataFrame(
        precision_recall_fscore_support(tfidf_predictions.actual,
                                        tfidf_predictions.predicted)).T
the_measures.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures.index = ['First', 'Third']
the_measures.accuracy = accuracy
print("tfidf measures")
print (the_measures)

accuracy2 = float(list(
        (count_predictions.actual == count_predictions.predicted)
        ).count(True))/float(len(count_predictions))
the_measures2 = pd.DataFrame(
        precision_recall_fscore_support(count_predictions.actual,
                                        count_predictions.predicted)).T
the_measures2.columns = ['precision', 'recall', 'fscore', 'accuracy']
the_measures2.index = ['First', 'Third']
the_measures2.accuracy = accuracy2
print("count measures")
print (the_measures2)

#%% Plots: Gender
sexlab = list(set((data.sex)))
#newlab = ["Trans MtF","Female", "Male", "Genderqueer", "T/M Couple", "Intersex", "M/F Couple"]
sexlab = [w.replace("Male and female couple", "M/F Couple") for w in sexlab]
sexlab = [w.replace("Transgender FtM","Trans FtM") for w in sexlab]
sexlab = [w.replace("Transgender MtF","Trans MtF") for w in sexlab]
sexlab = [w.replace("Transgender and male couple","T/M Couple") for w in sexlab]
sexlab = [w.replace("Transgender and female couple","T/F Couple") for w in sexlab]

sexfreq = []
for sex in sexlab:
    sexfreq.append(len(data[data.sex == sex]))

sexindex = list(range(len(sexlab)))

plt.bar(sexindex, sexfreq, color = ['gold', 'goldenrod', 'darkgoldenrod'])
plt.xlabel('Genders of Consumers')
plt.ylabel('Number of Consumers')
plt.title('Bar Plot of Gender Distribution Among Consumers')
plt.xticks(sexindex, sexlab, rotation = 30)
#plt.show()
plt.savefig("genderplot.png", transparent = True, bbox_inches = 'tight')

#%% Plots: Sexuality
sexualitylab = set((data.sexuality)).difference(set((data.sex)))

sexualityfreq = []
for lab in sexualitylab:
    sexualityfreq.append(len(data[data.sexuality == lab])) 

sexualityindex = list(range(len(sexualitylab)))

plt.bar(sexualityindex, sexualityfreq, color = ['gold', 'goldenrod', 'darkgoldenrod'])
plt.xlabel('Sexualities of Consumers')
plt.ylabel('Number of Consumers')
plt.title('Bar Plot of Sexuality Distribution Among Consumers')
plt.xticks(sexualityindex, sexualitylab, rotation = 30)
#plt.show()
plt.savefig("sexualityplot.png", transparent = True, bbox_inches = 'tight')

#%% Plots: Number of upvotes and downvotes per video
votelab = list(set((data.search)))

upavg = []
downavg = []
for q in votelab:
    isTrue = data["search"] == q
    Filter = data[isTrue]
    upavg.append(round(sum(Filter.upvotes)/len(Filter.upvotes)))
    downavg.append(round(sum(Filter.downvotes)/len(Filter.downvotes)))

pos = list(range(len(downavg)))
width = 0.25

fig, ax = plt.subplots(figsize = (10,4))

plt.bar(pos, downavg, width, alpha = 1, color = 'saddlebrown', label = votelab[0])
plt.bar([p + width for p in pos], upavg, width, alpha = 1, color = "peru", label = votelab[1])

ax.set_ylabel('Average number of votes')
ax.set_xticks([p + 1.5 * width for p in pos])
ax.set_xticklabels(votelab)
plt.xlim(min(pos)-width, max(pos)+width*3)
plt.ylim([0, max(upavg) + max(downavg)])

plt.legend(["down votes", "up votes"], loc = 'upper left')
plt.savefig("votesplot.png", transparent = True, bbox_inches = 'tight')

#%% Descriptive stats

print((len(data[data["search"] == "VR"]) + 
       len(data[data["search"] == "POV"]) + 
       len(data[data["search"] == "Cuckold"]) + 
       len(data[data["search"] == "Hidden_Cam"]))/4)

print(sum(data.upvotes[data["search"] == "VR"])/len(data.upvotes))
print(sum(data.upvotes[data["search"] == "POV"])/len(data.upvotes))
print(sum(data.upvotes[data["search"] == "Cuckold"])/len(data.upvotes))
print(sum(data.upvotes[data["search"] == "Hidden_Cam"])/len(data.upvotes))