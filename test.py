import requests

proxies = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050'
}

try:
    response = requests.get('http://httpbin.org/ip', proxies=proxies)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")
