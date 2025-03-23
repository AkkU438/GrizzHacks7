from bs4 import BeautifulSoup
import requests
import pandas as pd
import time

df = pd.read_csv("Data/current_NFL_RB.csv")
full_names = df["Player Name"]
#i = 1

i = 1
url = "https://www.sports-reference.com/cfb/players/derrick-henry-" + str(i) + ".html"
college_stats = requests.get(url)
soup = BeautifulSoup(college_stats.text, 'html.parser')
#print(soup)

position_check = soup.find('div', class_='nothumb')
if position_check:
    print("found the nothumb div.")
    position_tag = position_check.find('p', recursive=True)
    if position_tag and position_tag.find('strong') and "Position" in position_tag.find('strong').text:
        position = position_tag.text.split(":")[1].strip()
        print("Position: " + position)
        while position != "RB":
            time.sleep(10)
            i += 1
            url = "https://www.sports-reference.com/cfb/players/derrick-henry-" + str(i) + ".html"
            college_stats = requests.get(url)
            soup = BeautifulSoup(college_stats.text, 'html.parser')
            #print(soup)

            position_check = soup.find('div', class_='nothumb')
            if position_check:
                print("found the nothumb div.")
                position_tag = position_check.find('p', recursive=True)
                if position_tag and position_tag.find('strong') and "Position" in position_tag.find('strong').text:
                    position = position_tag.text.split(":")[1].strip()
                    print("Position: " + position)
    else:
        print("Position tag not found.")
else:
    print("no nothumb div found.")
#
#for full_name in full_names:
#
#    full_name = full_name.split(" ")
#    fName = full_name[0]
#    lName = full_name[1]
#    full_name = fName + lName
#
##    print(full_name)
##    print("first name " + fName + " last name " + lName)
##    print(i)
##    i += 1
#
#    i = 1
#    url = "https://www.sports-reference.com/cfb/players/" + fName + "-" + lName + "-" + str(i) + ".html"
#    college_stats = requests.get(url)
#    soup = BeautifulSoup(college_stats.text, 'html.parser')
#    #print(soup)
#
#    position_check = soup.find('div', class_='nothumb')
#    if position_check:
#        position_tag = position_check.find('p')
#        if position_tag and "Position" in position_tag.text:
#            position = position_tag.text.split(":")[1].strip()
#            print(position)
#            while position != "RB":
#                i += 1
#                url = "https://www.sports-reference.com/cfb/players/" + fName + "-" + lName + "-" + str(i) + ".html"
#                college_stats = requests.get(url)
#                soup = BeautifulSoup(college_stats.text, 'html.parser')
#
#                position_check = soup.find('div', class_='nothumb')
#                if position_check:
#                    position_tag = position_check.find('p')
#                    if position_tag and "Position" in position_tag.text:
#                        position = position_tag.text.split(":")[1].strip()
#                        print(position)
#