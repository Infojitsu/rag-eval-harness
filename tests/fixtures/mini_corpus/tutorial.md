# Brewing Tutorial

This tutorial walks through brewing your first cup with the Teapot API.

## Steps

First, start the teapot server. Then send a brew request:

```python
import requests

response = requests.post(
    "http://localhost:8418/brew",
    json={"tea": "green", "temperature_c": 80},
)
print(response.json()["status"])
```

The response contains a `brew_id` you can poll for status updates until the
tea is ready to pour.

## Errors

If you request coffee instead of tea, the server answers with HTTP status 418
("I'm a teapot") and does not start brewing anything at all.
