import requests
from bs4 import BeautifulSoup

def scrape_angelone_support_url(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    faq_tabs = soup.find_all('div', class_='tab')

    faq_data = []

    for tab in faq_tabs:
        tab_title = tab.find('label', class_='tab-label').text.strip() 
        tab_content = tab.find('div', class_='tab-content').text.strip()
        faq_data.append({
            "title": tab_title,
            "content": tab_content
        })

    return faq_data

print(scrape_angelone_support_url("https://www.angelone.in/support/add-and-withdraw-funds/quarterly-settlement-sebi-payout"))