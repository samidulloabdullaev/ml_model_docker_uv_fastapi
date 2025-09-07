import requests

url = "http://localhost:8080/predict"


curtomer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "yes",
    "tenure": 1,
    "phoneservice": "yes",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "yes",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "monthlycharges": 239.85,
    "totalcharges": 2129.85,
}

response = requests.post(url, json=curtomer)
churn = response.json()
print("response", churn)

if churn["churn"] == 1:
    print("Send email to the customer with promotion")
else:
    print("Do not send email to the customer")
