from bs4 import BeautifulSoup
import requests
import pandas as pd

all_information = []

page_url = 'https://www.fantasypros.com/nfl/reports/leaders/ppr-rb.php?year=2024'
fantasy_data = requests.get(page_url)
soup = BeautifulSoup(fantasy_data.text, 'html.parser')
rows = soup.find_all('tr')

for row in rows:
    player_name_tag = row.find('td', class_='player-label player-label-report-page')
    if player_name_tag:
        player_name = player_name_tag.text.strip()
        ttl_value_tag = row.find_all('td', class_='center')
        if len(ttl_value_tag) >= 1:
            ttl_value = ttl_value_tag[-1].text.strip()
            all_information.append([player_name, ttl_value])

df = pd.DataFrame(all_information, columns=['Player Name', 'TFP'])
df.to_csv("current_NFL_RB.csv", index=False)
print("CSV printed")