import pandas as pd
import gradio as gr
import joblib

# Load the saved Random Forest model and preprocessor
rf_model = joblib.load("best_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

def predict_customer_gender(country, state, product_category, sub_category, month, unit_price, customer_age):
    input_data = [[month, customer_age, country, state, product_category, sub_category, unit_price]]
    input_df = pd.DataFrame(input_data, columns=['Month', 'Customer Age', 'Country', 'State', 'Product Category', 'Sub Category', 'Unit Price'])

    # Transform the input data using the preprocessor
    input_data_transformed = preprocessor.transform(input_df)

    # Make gender predictions using the trained Random Forest model
    prediction = rf_model.predict(input_data_transformed)[0]
    
    # Display human readable output
    if prediction == 'M':
        return 'Male'
    else:
        return 'Female'


countries = ['United States', 'France', 'United Kingdom', 'Germany']

states = ['Washington', 'California', 'Oregon', 'Essonne', 'Yveline',
       'England', 'Hessen', 'Hamburg', 'Seine Saint Denis', 'Saarland',
       'Nordrhein-Westfalen', 'Bayern', 'Seine (Paris)', 'Pas de Calais',
       'Moselle', 'Hauts de Seine', 'Nord', 'Seine et Marne', 'Loiret',
       'Charente-Maritime', 'Loir et Cher', 'Brandenburg', 'Alabama',
       "Val d'Oise", 'Val de Marne', 'Minnesota', 'Wyoming', 'Ohio',
       'Garonne (Haute)', 'Kentucky', 'Texas', 'Missouri', 'Somme',
       'New York', 'Florida', 'Illinois', 'South Carolina',
       'North Carolina', 'Georgia', 'Virginia', 'Mississippi', 'Montana',
       'Arizona', 'Massachusetts', 'Utah']

product_categories = ['Accessories', 'Clothing', 'Bikes']

sub_category = ['Tires and Tubes', 'Gloves', 'Helmets', 'Bike Stands',
       'Mountain Bikes', 'Hydration Packs', 'Jerseys', 'Fenders',
       'Cleaners', 'Socks', 'Caps', 'Touring Bikes', 'Bottles and Cages',
       'Vests', 'Road Bikes', 'Bike Racks', 'Shorts']

months = ['February', 'March', 'April', 'June', 'July', 'August',
       'September', 'October', 'November', 'December', 'May', 'January']


# Define the input interface for the Gradio app
inputs = [
    gr.inputs.Dropdown(choices=countries, label="Country"),
    gr.inputs.Dropdown(choices=states, label="State"),
    gr.inputs.Dropdown(choices=product_categories, label="Product Category"),
    gr.inputs.Dropdown(choices=sub_category, label="Sub Category"),
    gr.inputs.Dropdown(choices=months, label="Month"),
    gr.inputs.Number(label="Unit Price"),
    gr.inputs.Number(label="Customer Age"),
]

app = gr.Interface(fn=predict_customer_gender, inputs=inputs, outputs="text", title="Product Release Planning App", live=True)

app.launch(share=True)