import requests

url = 'https://dataverse.harvard.edu/api/access/datafile/6180620?format=original'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
r = requests.get(url, headers=headers, stream=True, timeout=30)
print(f'Status: {r.status_code}')
print(f'Content-Type: {r.headers.get("Content-Type")}')
print(f'Content-Length: {r.headers.get("Content-Length")}')

if r.status_code == 200:
    with open('data/primekg/kg.csv', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print('Downloaded successfully')
else:
    print(f'Error: {r.text[:500]}')