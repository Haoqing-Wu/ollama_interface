import requests

url = 'http://127.0.0.1:8090/aigic'
response = requests.post(url, json={
    'debug': True,
    'prompt':"Explain the Newton's third law of motion. 100 words"
    # more parameters can be added here see below
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