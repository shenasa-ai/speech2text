from bs4 import BeautifulSoup
import requests
import numpy as np
import urllib.request
from urllib.error import HTTPError
import os


MAIN_URL = "http://radio.iranseda.ir/"
TOTAL_LINKS_ARRAY = np.array([])
# ==========================   CHANGE THIS NAME  ==========================================
Program_Name = 'Payam_Shamgahi' # RadioNAme_ProgrameName(i)

def find_episodes_urls_in_main_program_page(URL):
    global TOTAL_LINKS_ARRAY
    r = requests.get(URL)
    r.encoding = 'utf-8'
    soup = BeautifulSoup(r.text,"html.parser") 

    episodes = soup.find_all("a", {"class": "ch-text-color page-loding"}, href=True)
    i = 0
    for episode in episodes :
        episode_href = episode.get('href')
        TOTAL_LINKS_ARRAY = np.append(TOTAL_LINKS_ARRAY, find_episode_files(episode_href))
    return TOTAL_LINKS_ARRAY

def find_episode_files(EPISODE_URL):
    my_episode_request = requests.get(MAIN_URL + EPISODE_URL[3:])
    my_episode_request.encoding = 'utf-8'
    my_soup = BeautifulSoup(my_episode_request.text,"html.parser") 
    modal_bodies = my_soup.find_all("div", {"class": "modal-body"}) 
    # print(modal_bodies)
    array = np.array([])
    for modal_body in modal_bodies:
        array = np.append(array, find_proper_episode_file_link_to_download(modal_body))
    return array


def find_proper_episode_file_link_to_download(modal_body):
    link = ''
    for children in modal_body.findChildren( recursive=False) : 
        # if children.find('span').getText().find('دانلود از سرور 7') != -1:
        #     continue
        if ('mp4' in children.find('span').getText()):
            link  = children.find('a')['href']
            break
    return link

def download_Files(array):
    os.mkdir('./' + Program_Name)
    for i, file_url in enumerate(array):
        print(f'I am downloading {file_url}, this is number {i}')
        file_name = './' + Program_Name + '/' + Program_Name + str(i) + '.mp4'
        # urllib.request.urlretrieve(file_url, file_name)
        if file_url != '':
            try : 
                urllib.request.urlretrieve(file_url, file_name)
            except HTTPError as e:
                print(f' problem link is : {file_url} and the number is {i}')
# ==========================   CHANGE THIS url  ==========================================
find_episodes_urls_in_main_program_page('http://radio.iranseda.ir/Program/?VALID=TRUE&ch=12&m=024104')
download_Files(TOTAL_LINKS_ARRAY)

