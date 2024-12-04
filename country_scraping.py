import os
import os.path as op
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests as req
from bs4 import BeautifulSoup as Bs
from tqdm.notebook import tqdm


from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googlesearch import search
from waybackpy import WaybackMachineCDXServerAPI
from waybackpy.exceptions import NoCDXRecordFound
#from concurrent.futures import ThreadPoolExecutor, as_completed



socialblade_url = r"https://socialblade.com/youtube/user/"
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
API_KEY = pd.read_json(op.join("..", "config.json"))["api_key"][0]  # local file w personal API key


# ______________________________________________________________________________________________________________________
# Functions to extract countries from the education channels - YouTube API
# ______________________________________________________________________________________________________________________

def extract_channels_edu(path_edu, N_BATCHES, verbose=False):
    channels = []
    for i in range(N_BATCHES):
        if verbose:
            print(f"Processing file : path_edu_{i}", end="")
        edu = pd.read_csv(path_edu.format(i), index_col=0)
        ch = list(pd.unique(edu["channel_id"]))
        if verbose:
            print(f"  --> Found {len(ch)} channels")
        channels.extend(ch)
    channels = list(set(channels))  # take unique of the junction
    if verbose:
        print("Total number of unique channels :", len(channels))
    return channels


def agglomerate_countries(x, val_counts, filter=10):
    if type(x) == str and val_counts[x] < filter:
        return "Other"
    elif type(x) == str and x == "deleted":  # assign deleted to 'unknown'
        return "?"
    elif type(x) == float:  # assign NaN to 'unknown'
        return "?"
    else:
        return x




def extract_countries_socialblade(code=None, user=None, username=False, retries=5, verbose=True):
    if username: # first solution to get the webarchive page from the guessed username/handle since we couldn't get it from the Youtube API for deleted channels
        user_url = socialblade_url + user # try to guess the username with channel name
    else: # querries the website for the user code, it redirects us to the correct page
        user_url = 'http://socialblade.com/youtube/s/?q=' + code 
        user = code
    
    try:
        cdx_api = WaybackMachineCDXServerAPI(user_url, user_agent)
        #latest_url = cdx_api.newest().archive_url
        crawl_url = cdx_api.near(year=2019, month=10, day =30).archive_url # use at time close to crawl date
        print(crawl_url)
        # Attempt to get the url a certain number of retries
        r = None
        for _ in range(retries):
            try:
                r = req.get(crawl_url, timeout=30)
                break # if successful do not retry
            except req.exceptions.ConnectionError as e:
                # if there is a connection error try again a bit later
                print('Reattempting ...', e) 
                time.sleep(4) # seconds
            except Exception as e:
                print('Error with the request', e)
                return 'request error'
        if r:
            if r.status_code == 200:
                soup = Bs(r.text, 'html.parser')
                country_search = soup.find('span', {'id': 'youtube-stats-header-country'})
                #handle = soup.find('div', {'id': 'fav-bubble'}).get('title', 'unknown_handle').split(' ')[-1] # assume it will always return something
                
                if country_search is None:
                    return 'no_country' # the user probably did not indicate a country
                else :
                    country = country_search.text
                    if verbose: tqdm.write(f"{user} : {country}                                         ")
                    return country
            else:
                return 'wrong status code'
        else:
            return 'attempts exhausted'
        
    except NoCDXRecordFound:
        tqdm.write(f"No record found for user {user}                       ", end="\r")
        return 'no_rec'
    except req.exceptions.ConnectionError as e:
        # these must be reattempted since the problem is in connecting to the API
        print(e)
        return 'api'
    
    except Exception as e:
        print('Other General exception', e)
        return 'except'






# Youtube API___________________________________________________________________________________________________________

def youtube_country_scraper(channel_ids, verbose=False):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    youtube = build("youtube", "v3", developerKey=API_KEY)
    if type(channel_ids) == list:
        ids_string = ",".join(channel_ids)
        countries = {ch: "no_result" for ch in channel_ids}
    elif type(channel_ids) == str:
        ids_string = channel_ids
        countries = {channel_ids : 'no_result'}

    request = youtube.channels().list(part="snippet", id=ids_string)
    items = request.execute()
    if "items" in items:  # for when you redo with single channels
        for item in items.get("items", []):
            if "snippet" in item:
                id = item.get("id")
                country = item.get("snippet").get("country")
                if (
                    id in channel_ids
                ):  # else the channel now has a different id and need to be redone
                    countries[id] = country
    if verbose:
        print(items)
        print(countries)
    return countries


def youtube_handle_scraper(channel_ids, verbose=False):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    youtube = build("youtube", "v3", developerKey=API_KEY)
    if type(channel_ids) == list:
        ids_string = ",".join(channel_ids)
        handles = {ch: "no_result" for ch in channel_ids}
    elif type(channel_ids) == str:
        ids_string = channel_ids
        handles = {channel_ids : 'no_result'}

    request = youtube.channels().list(part="snippet", id=ids_string)
    items = request.execute()
    if "items" in items:  # for when you redo with single channels
        for item in items.get("items", []):
            if "snippet" in item:
                id = item.get("id")
                handle = item.get("snippet").get("customUrl")
                if (id in channel_ids):  # else the channel now has a different id and need to be redone
                    handles[id] = handle 
    if verbose:
        print(items)
        print(handles)
    return handles



def get_youtube_username(channel_id): #DID  NOT WORK
    query = f"{channel_id} site:youtube.com"
    for url in search(query, num_results=10):  # Search Google for the channel
        if "youtube.com/channel/" in url:  # Look for the YouTube channel link
            print(f"Found YouTube channel URL: {url}")
            response = req.get(url)
            if response.status_code == 200:
                soup = Bs(response.text, "html.parser")
                f = open("soup.txt", "a")
                f.write(soup.prettify())
                f.close()
                # Find the username in the title or meta tags
                span_element = soup.find("span", class_="yt-core-attributed-string yt-content-metadata-view-model-wiz__metadata-text yt-core-attributed-string--white-space-pre-wrap yt-core-attributed-string--link-inherit-color")

                # Extract the text
                if span_element:
                    text = span_element.get_text(strip=True)
                    if text.startswith("@"):
                        username = text[1:]  # Remove '@' if needed
                        print(f"Username: {username}")
                    else:
                        print("No '@' found in the text.")
                else:
                    print("Span element not found.")
                    return soup
    return None
