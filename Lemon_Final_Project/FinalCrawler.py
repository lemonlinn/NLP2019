# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:35:37 2019

@author: Lemon Lin Reimer

@note: this program requires selenium, chrome browser, and chromedriver to run
@note: chromedriver must either be placed in PATH or specify the path in init
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd

chromeoptions = Options()
chromeoptions.add_argument("--headless")

class mycrawler(object):
    
    def __init__(self):
        """initializes the driver"""
        self.driver = webdriver.Chrome(chrome_options=chromeoptions)
        print('Driver has been launched in headless mode.')
  
    def teardown(self):
        """closes the driver"""
        self.driver.quit()
        print('Driver has been closed.')
    
    def links(self, query):
        """searches query and extracts links from first page of videos (36)"""
        print('Retrieving links for ' + query + "...")
        self.driver.get('https://www.xhamster.com/')
        search_trigger = self.driver.find_element_by_xpath('//*[@action="https://xhamster.com/search"]')
        search_trigger.click()
        search_bar = search_trigger.find_element_by_xpath('//*[@class="search-text"]')
        search_bar.clear()
        search_bar.send_keys(query)
        submit = self.driver.find_element_by_xpath('//*[@class="search-submit"]')
        submit.click()
        links = []
        
        content = self.driver.find_elements_by_xpath('//*[@class="thumb-list__item video-thumb"]')
        for i in content:
            vid_id = i.find_element_by_xpath('.//*[@class="video-thumb__image-container thumb-image-container"]')
            links.append(vid_id.get_attribute("href"))
        return(links) 
        
    def get_pos(self, word):
        import nltk
        from nltk.corpus import wordnet
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    
    def comments(self, query):
        """scrapes information from every comment in the links and places in pandas dataframe"""
        links = self.links(query)
        print('Retrieving comments for ' + query + "...")
        tmpDF = pd.DataFrame()
        cnt = 0
        for link in links:
            cnt += 1
            try:
                self.driver.get(link)
                print("link " + str(cnt) + " out of " + str(len(links)))
                viewcount = self.driver.find_element_by_xpath('//div[@class="header-icons"]/span').text
                #title = self.driver.find_element_by_xpath('//div[@class="width-wrap with-player-container"]/span').text
                allvotes = self.driver.find_element_by_xpath('//div[@class="rb-new__info"]').text
                upvote = allvotes.split(' / ')[0]
                downvote = allvotes.split(' / ')[-1]
                allcomments = self.driver.find_element_by_xpath('//*[@class="comments-list"]')
                comment_list = allcomments.find_elements_by_xpath('.//*[@class="item"]')
                comment = []
                gender = []
                user = []
                date = []
                location = []
                for c in comment_list:
                    tmp_comment = c.find_element_by_xpath('.//div[2]/div[1]/div[3]/span[1]')
                    #tmp_comment = c.find_element_by_class_name("comment-info").\
                    #find_element_by_class_name("comment-body").\
                    #find_element_by_class_name("text").\
                    #find_element_by_class_name("comment-text")
                    comment.append(tmp_comment.text)
                
                    tmp_gender = c.find_element_by_xpath('.//div[@class="sex"]/i')
                    gender.append(tmp_gender.get_attribute("data-tooltip"))
                
                    tmp_user = c.find_element_by_xpath('.//div[1]/a')
                    user.append(tmp_user.get_attribute("href"))
                
                    tmp_date = c.find_element_by_xpath('.//div[2]/div[1]/div[4]/div[1]')
                    date.append(tmp_date.get_attribute("data-tooltip"))
                
                    tmp_loc = c.find_element_by_xpath('.//div[@class="user-info"]/div[1]')
                    location.append(tmp_loc.get_attribute("data-tooltip"))
                
                    new_query = query.split(" ")
                    new_query = "_".join(new_query)
        
                for i,c in enumerate(comment):
                    tmpDict = {'comment':c,'gender':gender[i], 'location':location[i], 'user':user[i],'date':date[i],'video':link, 'views':viewcount, 'upvotes':upvote, 'downvotes':downvote, "search":new_query}
                    tmpDF = tmpDF.append(tmpDict, ignore_index = True)
            
            except:
                print("Uh oh, something went wrong! Moving to next link...")
                
        return(tmpDF)
        
    def cleaner(self, data):
        """Cleans data and appends new columns"""
        import re
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        print('Comments are being cleaned...')
        
        clean = []

        for com in data['comment']:
            try:
                clean.append(re.sub('[^A-Za-z:]+'," ",re.sub('http\S+',"",com)))
            except:
                clean.append(None)
            
        pattern = re.compile(".*?:$")

        for i,c in enumerate(clean):
            try:
                if pattern.match(c):
                    clean[i] = None
            except:
                pass
        
        data["cleancomment"] = clean

        data = data[data['cleancomment'].isna() == False]
        data = data[data['cleancomment'] != ""]
        
        tmp2 = []
        tmp3 = []

        for sex in data.gender:
            tmp = re.split(", ",sex)
            tmp2.append(tmp[0])
            tmp3.append(tmp[-1])

        data["sex"] = tmp2
        data["sexuality"] = tmp3

        data = data[data.sex != ""]
        data = data[data.sexuality != "Male"]
        
        newup = []
        newdown = []
        newviews = []

        for i in data.upvotes:
            newup.append(int(re.sub('[^0-9]+', "", i)))
        for j in data.downvotes:
            newdown.append(int(re.sub('[^0-9]+', "", j)))
        for k in data.views:
            newviews.append(int(re.sub('[^0-9]+', "", k)))
    
        data['downvotes'] = newdown  
        data['upvotes'] = newup
        data['views'] = newviews

        stops = stopwords.words('english')
        stemmer = PorterStemmer()
        lemmer = WordNetLemmatizer()

        lemma = []
        stem = []

        for com in data.cleancomment.str.lower():
            tokenwrd = word_tokenize(com)
            tmp3 = [stemmer.stem(wrd) for wrd in tokenwrd if wrd not in stops]
            tmp4 = [lemmer.lemmatize(wrd, self.get_pos(wrd)) for wrd in tokenwrd if wrd not in stops]
            stem.append(tmp3)
            lemma.append(tmp4)
    
        data["stem"] = stem
        data["lemma"] = lemma
        
        return(data)