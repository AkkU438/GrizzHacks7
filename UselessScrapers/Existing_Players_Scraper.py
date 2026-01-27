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
        
        player_url_tag = player_name_tag.find('a')
        if player_url_tag:
            player_url = player_url_tag['href']

            player_data = requests.get("https://www.fantasypros.com" + player_url)
            player_soup = BeautifulSoup(player_data.text, 'html.parser')

            bio_section = player_soup.find('div', class_='bio')
            if bio_section:
                bio_details = {span.text.split(':')[0].strip(): span.text.split(':')[1].strip() for span in bio_section.find_all('span', class_='bio-detail')}
                height = bio_details.get('Height', 'N/A')
                weight = bio_details.get('Weight', 'N/A')
                age = bio_details.get('Age', 'N/A')
                college = bio_details.get('College', 'N/A')
            else:
                height, weight, age, college = 'N/A', 'N/A', 'N/A', 'N/A'
        else:
            height, weight, age, college = 'N/A', 'N/A', 'N/A', 'N/A'

        ttl_value_tag = row.find_all('td', class_='center')
        if len(ttl_value_tag) >= 1:
            ttl_value = ttl_value_tag[-1].text.strip()

        all_information.append([player_name, ttl_value, height, weight, age, college])

df = pd.DataFrame(all_information, columns=['Player Name', 'TFP', 'Height', 'Weight', 'Age', 'College'])
df.to_csv("Data/current_NFL_RB.csv", index=False)
print("CSV printed")