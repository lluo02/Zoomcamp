#!/usr/bin/env python
# coding: utf-8


import requests


url = 'http://localhost:9696/predict'


customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85
}


response = requests.post(url, json=customer).json()
response


if response['churn'] == True:
    print("sending promo email to %s" % ('xyz-123'))
else:
    print("not sending promo email to %s" % ('xyz-123'))





