from bs4 import BeautifulSoup
import time
import requests
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in the background
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

def get_player_names(url):
    driver.get(url)
    driver.implicitly_wait(5)
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    players = soup.find_all('h4', class_='team-meta__name')
    prospect_names = []
    for player in players:
        first_name = player.find('span', class_='firstName').text.strip()
        last_name = player.find('span', class_='lastName').text.strip()
        print(first_name, last_name)
        full_name = f"{first_name} {last_name}"
        prospect_names.append(full_name)
    return prospect_names    

all_prospect_names = []
for page_name in range(1,6):
    url = f'https://www.nfldraftbuzz.com/positions/RB/{page_name}/2025'
    prospect_names = get_player_names(url)
    all_prospect_names.extend(prospect_names)
df = pd.DataFrame(all_prospect_names, columns=['Player Name'])
df.to_csv("nfl_draft_prospects.csv", index=False)
driver.quit()
print("CSV printed")
