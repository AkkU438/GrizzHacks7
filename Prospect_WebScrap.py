from bs4 import BeautifulSoup
import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

# Set up Selenium WebDriver
chrome_options = Options()
chrome_options.add_argument("--headless")  # Run in headless mode
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)
driver.set_page_load_timeout(180)  # Set a longer timeout if necessary
driver.set_script_timeout(180)  # Adjust the script timeout as well

def get_player_details(player_url):
    """
    Given a player's individual page URL, scrape details like Height, Weight, and College.
    """
    driver.get(player_url)
    time.sleep(20)  # Allow time for content to load
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    player_info = {
        'Height': 'N/A',
        'Weight': 'N/A',
        'College': 'N/A'
    }
    
    # These containers hold each stat item (Height, Weight, College, etc.)
    player_stats = soup.find_all('div', class_='player-info-details__item')

    print(f"Scraping details from: {player_url}")

    if not player_stats:
        print("No player details found! Check the page structure.")
        return player_info

    for stat in player_stats:
        # FIX: Corrected the typo here from 'player-infor-details__title' to 'player-info-details__title'
        title = stat.find('h6', class_='player-info-details__title')
        value = stat.find('div', class_='player-info-details__value')

        if title and value:
            title_text = title.text.strip()
            value_text = value.text.strip()

            if title_text == "Height":
                player_info['Height'] = value_text
            elif title_text == "Weight":
                player_info['Weight'] = value_text
            elif title_text == "College":
                player_info['College'] = value_text

    print(f"Extracted: {player_info}")
    return player_info

def get_player_names(url):
    """
    Scrape the main page table for a list of players (names + link to their detail page).
    """
    driver.get(url)
    
    # Wait for the page to load
    WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located((By.CLASS_NAME, 'team-meta__name'))
    )
    time.sleep(3)  # Give JavaScript extra time to load if needed
    
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    players = soup.find_all('h4', class_='team-meta__name')

    print(f"Scraping {url} - Found {len(players)} players.")
    
    if not players:
        print("No players found! Check the website structure.")
        return []

    prospect_names = []
    
    for player in players:
        first_name_tag = player.find('span', class_='firstName')
        last_name_tag = player.find('span', class_='lastName')
        
        # Skip if either span is missing
        if not first_name_tag or not last_name_tag:
            print("Skipping a player due to missing name tag.")
            continue
        
        first_name = first_name_tag.text.strip()
        last_name = last_name_tag.text.strip()
        full_name = f"{first_name} {last_name}"
        
        # Attempt to find the player's individual page link
        row = player.find_parent('tr')  # Move up to the <tr> element
        player_link_tag = row.find('a', href=True) if row else None
        
        if player_link_tag:
            player_url = "https://www.nfldraftbuzz.com" + player_link_tag['href']
            print(f"Fetching details for {full_name} -> {player_url}")
            player_data = get_player_details(player_url)
            
            # Append data to our list
            prospect_names.append([full_name,
                                   player_data['Height'],
                                   player_data['Weight'],
                                   player_data['College']])
        else:
            print(f"Skipping {full_name}, no player URL found.")
    
    print(f"Collected {len(prospect_names)} players from this page.")
    return prospect_names


# ------------------- Main Script -------------------

all_prospect_names = []
# Adjust the range if you want to scrape more or fewer pages
for page_num in range(1, 6):
    url = f'https://www.nfldraftbuzz.com/positions/RB/{page_num}/2025'
    prospect_names = get_player_names(url)
    
    if prospect_names:
        all_prospect_names.extend(prospect_names)
    print(f"Total prospects collected so far: {len(all_prospect_names)}\n")

# Final check and DataFrame creation
if not all_prospect_names:
    print("No data was scraped. Check website structure and class names.")
else:
    df = pd.DataFrame(all_prospect_names, columns=['Player Name', 'Height', 'Weight', 'College'])
    
    print("Sample Data:")
    print(df.head())  # Show a few rows
    
    df.to_csv("nfl_draft_prospects_fixed.csv", index=False)
    print("CSV saved successfully!")

driver.quit()
