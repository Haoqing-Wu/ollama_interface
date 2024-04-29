import requests

url = 'http://127.0.0.1:8090/aigic'
response = requests.post(url, json={
    'debug': True,
    'prompt':"who are you?"
    })

if response.status_code == 200:
    print('String sent successfully!')
    result = response.json()
    content = result['content']['output']
    print('Feedback result:', content)
else:
    print('Error:', response.status_code)
    print('Reason:', response.reason)
    print('Content:', response.content)