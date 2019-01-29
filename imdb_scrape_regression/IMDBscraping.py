
# Goal: Scrape information with BeautifulSoup from top 10,000 box office grossing movies on imdb.com



# =============================================================================
# Function to grab URLs
# =============================================================================

def imdb_urls():
    '''
    Grabs all urls and returns them as a list. 
    This list will be the input to the scraping function.
    '''
    
    imdb_url_list = []
    page_num = 1
    
    # this imdb sort page shows 50 movies at a time, +50 in counter advances to next page
    # e.g., page_num of 901 shows the 901-950th ranked top grossing movies
    while page_num < 10001:
        imdb_url_list.append('https://www.imdb.com/search/title?title_type=feature&sort=boxoffice_gross_us,desc&start='+str(page_num)+'&ref_=adv_nxt')                    
        page_num +=50
        print(page_num)

    return imdb_url_list


# =============================================================================
# PICKLING AREA
# (This area is simply to save the output of the functions)
# =============================================================================
import pickle

# Gets the URLs!
imdb_top10k_urls = imdb_urls()

# save url list
with open('imdb_urls.pkl', 'wb') as picklefile:
    pickle.dump(imdb_top10k_urls, picklefile)                     
# open url list
with open("imdb_urls.pkl", 'rb') as picklefile: 
    imdb_top10k_urls_p = pickle.load(picklefile)   


# pass in url list to scraper function
imdb_full = imdb_scraper(imdb_top10k_urls_p)

# pickle full data file
with open('imdb_full_data.pkl', 'wb') as picklefile:
    pickle.dump(imdb_full, picklefile)                     
# open pickle save
with open("imdb_full_data.pkl", 'rb') as picklefile: 
    imdb_full_data_p = pickle.load(picklefile)  




# =============================================================================
# Main scraper function
# =============================================================================

def imdb_scraper(imdb_urls):
    '''Scrapes most information from IMDb top 10,000 grossing feature films.
    1)IMDb ranking, 2)title, 3)year of release, 4)MPAA rating, 5)runtime,
    6)genre, 7)IMDb rating, 8)director, 9)number of votes, and 10) domestic gross
    '''
    
    import requests
    requests.__path__ 
    from bs4 import BeautifulSoup
    import re
    import time
    import random
    
    imdb_headers = ['imdb_ranking','imdb_title','imdb_year','imdb_MPAA','imdb_runtime',
                        'imdb_genre','imdb_rating','imdb_metascore','imdb_director','imdb_num_votes','imdb_gross']
        
    imdb_data = []
    
    for page in imdb_urls:
        
        imdb_url = page
        response5 = requests.get(imdb_url)
        page5 = response5.text
        soup5 = BeautifulSoup(page5, 'lxml')
               
        movies_on_page = soup5.find_all(class_="lister-item-content")
        
        # Scrapes info from each of the 50 movies shown on given page
        # Does some cleaning along the way
        for movie in movies_on_page:    #  soup5.find_all(class_="lister-item mode-advanced"):
            
            imdb_irank = movie.find(class_="lister-item-index").text.replace(".","")
            imdb_title = movie.find(class_="lister-item-header").a.text
            imdb_year = movie.find(class_="lister-item-year").text
            
            if not movie.find(class_="certificate"):
                imdb_mpaa = "N/A"
            else:
                imdb_mpaa = movie.find(class_="certificate").text
                
            if not movie.find(class_="runtime"):
                imdb_runtime = "N/A"
            else:
                imdb_runtime = movie.find(class_="runtime").text.replace(" min","")
                
            if not movie.find(class_="genre"):
                imdb_genre = "N/A"
            else:
                imdb_genre = movie.find(class_="genre").text.replace("\n","")
                
            if not movie.find(class_="ratings-imdb-rating"):
                imdb_rating = "N/A"
            else:
                imdb_rating = movie.find(class_="ratings-imdb-rating").text.replace("\n","")
            
            if not movie.find(class_="ratings-metascore"):
                imdb_metascore = "N/A"
            else:
                imdb_metascore = movie.find(class_="ratings-metascore").text.replace("\n","").replace("Metascore","").strip()
            
            # note: if multiple directors only scrapes first listed
            if not movie.find(text=re.compile("   Director")):
                imdb_director = "N/A"
            elif not movie.find(text=re.compile("   Director")).nextSibling:
                imdb_director = "N/A"
            else:
                imdb_director = movie.find(text=re.compile("   Director")).nextSibling.text
          
            
            # Split value in two at "|" (this separates rating count and gross on website)
            if not movie.find(class_="sort-num_votes-visible"):
                imdb_rating_count = "N/A"
                imdb_gross = "N/A"
            elif "Votes" not in movie.find(class_="sort-num_votes-visible").text:
                imdb_rating_count = "N/A"
                imdb_gross = "N/A"
            else:
                imdb_count_gross = movie.find(class_="sort-num_votes-visible").text.split('|') #.replace("\n","")
                imdb_rating_count = imdb_count_gross[0].replace("\n","").replace("Votes:","").replace(",","")    
                # gross earnings are in Millions (e.g., 54.84 is 54,840,000)
                imdb_gross = imdb_count_gross[1].replace("Gross:","").replace("\n","").replace("$","").replace("M","").strip()
                 
            
            imdb_page_dict = dict(zip(imdb_headers, [imdb_irank, imdb_title, imdb_year, imdb_mpaa, imdb_runtime,
                                                imdb_genre, imdb_rating, imdb_metascore, imdb_director, imdb_rating_count, imdb_gross]))
            
            imdb_data.append(imdb_page_dict)
            
            print(imdb_director)
            
        print(imdb_irank)
        time.sleep(.5+2*random.random())
    
    return (imdb_data)





























