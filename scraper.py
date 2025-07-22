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
            "content": tab_content,
            "url": url
        })
    return faq_data

def get_all_angelone_support_urls():
    response = requests.get("https://www.angelone.in/support")
    soup = BeautifulSoup(response.text, 'html.parser')
    categories =  soup.find('div', class_='cat-grid').find_all('div', class_='grid')

    for category in categories:
        category_url = category.find('a')['href']

        resp = requests.get(category_url)
        soup = BeautifulSoup(resp.text, 'html.parser')

        subcategories = soup.find('div', class_='list-item').find_all('a')

        for subcategory in subcategories:
            url = subcategory['href']
            faq_data = scrape_angelone_support_url(url)
            yield faq_data
