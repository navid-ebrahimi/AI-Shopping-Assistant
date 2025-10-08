# import requests

# url = "https://model-api.darkube.app/embed"
# data = {"texts": "سلام دنیا"}
# res = requests.post(url, json=data)
# print(res.json())

import requests

url = "https://model-api.darkube.app/health"

response = requests.get(url)

print("Status code:", response.status_code)
print("Response JSON:", response.json())
