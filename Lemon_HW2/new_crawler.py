# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 13:05:36 2019

@author: lemon
"""

class crawler(object):

    def my_scraper(self, tmp_url_in):
        from bs4 import BeautifulSoup
        import requests
        import re
        from nltk.corpus import stopwords
        sw = stopwords.words('english')
        tmp_text = ''
        try:
            content = requests.get(tmp_url_in)
            soup = BeautifulSoup(content.text, 'html.parser')
    
            tmp_text = soup.findAll('p') 
    
            tmp_text = [word.text for word in tmp_text]
            tmp_text = [word for word in tmp_text if word not in sw]
            tmp_text = ' '.join(tmp_text)
            tmp_text = re.sub('\W+', ' ', re.sub('xa0', ' ', tmp_text))
            tmp_text = re.sub(r'\w*\d\w*','',tmp_text).strip()
        except:
            pass
    
        return tmp_text
    
    def fetch_urls(self, query, cnt):
        #now lets use the following function that returns
        #URLs from an arbitrary regex crawl form google
    
        #pip install pyyaml ua-parser user-agents fake-useragent
        import requests
        from fake_useragent import UserAgent
        from bs4 import BeautifulSoup
        import re 
        ua = UserAgent()
    
        google_url = "https://www.google.com/search?q=" + query + "&num=" + str(cnt)
        response = requests.get(google_url, {"User-Agent": ua.random})
        soup = BeautifulSoup(response.text, "html.parser")
    
        result_div = soup.find_all('div', attrs = {'class': 'ZINbbc'})
    
        links = []
        titles = []
        descriptions = []
        for r in result_div:
            # Checks if each element is present, else, raise exception
            try:
                link = r.find('a', href = True)
                title = r.find('div', attrs={'class':'vvjwJb'}).get_text()
                description = r.find('div', attrs={'class':'s3v9rd'}).get_text()
    
                # Check to make sure everything is present before appending
                if link != '' and title != '' and description != '': 
                    links.append(link['href'])
                    titles.append(title)
                    descriptions.append(description)
            # Next loop if one element is not present
            except:
                continue  
    
        to_remove = []
        clean_links = []
        for i, l in enumerate(links):
            clean = re.search('\/url\?q\=(.*)\&sa',l)
    
            # Anything that doesn't fit the above pattern will be removed
            if clean is None:
                to_remove.append(i)
                continue
            clean_links.append(clean.group(1))
    
        # Remove the corresponding titles & descriptions
        for x in to_remove:
            del titles[x]
            del descriptions[x]
            
        return clean_links
 
    def write_crawl_results(self, my_query, the_cnt_in):
        #let use fetch_urls to get URLs then pass to the my_scraper function 
        import pandas as pd
        from nltk.stem import PorterStemmer
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        
        global the_data
        the_data = pd.DataFrame(columns=['body_basic','body_lemma','body_stem','label'])
        the_urls_list = self.fetch_urls(my_query, the_cnt_in)
        
        cnt = 0
        for word in the_urls_list:
            tmp_txt = self.my_scraper(word)
            porter = PorterStemmer()
            lemmer = WordNetLemmatizer()
            
            if len(tmp_txt) != 0:
                try:
                    tokenwrd = word_tokenize(tmp_txt)
                    stemsent = " ".join([porter.stem(i) for i in tokenwrd])
                    lemmasent = " ".join([lemmer.lemmatize(i) for i in tokenwrd])
                    new_query = my_query.split(" ")
                    new_query = "_".join(new_query)
                    print(new_query)
                    the_data.loc[cnt] = [tmp_txt] + [lemmasent] + [stemsent] + [new_query]
                    print (word)
                    cnt += 1
                except:
                    pass
        return(the_data)