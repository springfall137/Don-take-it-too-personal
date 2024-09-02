import requests
from bs4 import BeautifulSoup

데이터 = requests.get('https://finance.naver.com/item/sise.nhn?code=005930')

soup = BeautifulSoup(데이터.content, 'html.parser')

# print(데이터.content)
# print(데이터.status_code)
print(soup.find_all('strong', id="_nowVal")[0].text)
print(soup.find_all('span', class_="tah")[5].text)
