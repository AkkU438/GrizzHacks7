from bs4 import BeautifulSoup
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


url = 'https://www.nfldraftbuzz.com/positions/RB/1/2025'

driver.get(url)
driver.implicitly_wait(5)

soup = BeautifulSoup(driver.page_source, 'html.parser')

players = soup.find_all('h4', class_='team-meta__name')



prospect_names = []
for player in players:
    first_name = player.find('span', class_='firstName').text.strip()
    last_name = player.find('span', class_='lastName').text.strip()
    print(first_name, last_name)
    full_name = f"{first_name} {last_name}"
    prospect_names.append(full_name)

print(prospect_names)

driver.quit()

df = pd.DataFrame(prospect_names, columns=['Player Name'])
df.to_csv("nfl_draft_prospects.csv", index=False)
print("CSV printed")