import requests
import json

url = 'http://127.0.0.1:5001/predictions'
payload = json.dumps({
  "input": {
    "debug": True,
    "prompt": "Who are you?",
    "mirostat": 0,
    "mirostat_eta": 0.1,
    "mirostat_tau": 5.0,
    "num_ctx": 2048,
    "repeat_last_n": 64,
    "repeat_penalty": 1.1,
    "temperature": 0.8,
    "seed": 0,
    "stop": ["\n"],
    "tfs_z": 1,
    "num_predict": 128,
    "top_k": 40,
    "top_p": 0.9
  }
})
headers = {
  'Content-Type': 'application/json',
  'Accept': 'application/json'
}
response = requests.request("POST", url, headers=headers, data=payload)

if response.status_code == 200:
    print('String sent successfully!')
    result = response.json()
    content = result['content']['output']
    print('Feedback result:', content)
else:
    print('Error:', response.status_code)
    print('Reason:', response.reason)
    print('Content:', response.content)

'''

possible parameters(default), see https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

class InputData(BaseModel):
    debug: bool = True
    prompt: str
    mirostat: int = 0
    mirostat_eta: float = 0.1
    mirostat_tau: float = 5.0
    num_ctx: int = 2048
    repeat_last_n: int = 64
    repeat_penalty: float = 1.1
    temperature: float = 0.8
    seed: int = 0
    stop: list = ["\n"]
    tfs_z: int = 1
    num_predict: int = 128
    top_k: int = 40
    top_p: float = 0.9
'''