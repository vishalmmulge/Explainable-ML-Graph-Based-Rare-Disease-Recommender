import requests
import json

# Get dataset metadata from Dataverse
doi = '10.7910/DVN/IXA7BM'
url = f'https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId=doi:{doi}'
headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
print(f'Status: {response.status_code}')
if response.status_code == 200:
    data = response.json()
    files = data.get('data', {}).get('latestVersion', {}).get('files', [])
    for f in files:
        print(f"{f['dataFile']['filename']}: {f['dataFile']['filesize']} bytes - ID: {f['dataFile']['id']}")