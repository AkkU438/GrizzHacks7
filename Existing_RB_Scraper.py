from bs4 import BeautifulSoup
import requests

url = "https://www.espn.com/nfl/players/_/position/rb"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive"
}

page = requests.get(url, headers = headers)
soup = BeautifulSoup(page.text, 'html.parser')

#print(soup.prettify())
tables = soup.find_all('table', class_ = 'tablehead')

for table in tables:
    print(table.prettify())
