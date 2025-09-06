import pickle

example = {
    'customerid': '7590-vhveg',
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'yes',
    'tenure': 1,
    'phoneservice': 'yes',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'yes',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'monthlycharges': 239.85,
    'totalcharges': 2129.85,
}

# read the model
with open('model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

def predict(features):

    preds = model.predict_proba(features)

    if preds[0, 1] >= 0.5:
        message = 'Churn'
    else:
        message = 'No Churn'
    
    result = {
        'churn_probability': float(preds[0, 1]),
        'churn': message
    }

    return result

if __name__ == '__main__':
    print(predict(example))