import pandas as pd
import gradio as gr
import joblib

# Load the saved Random Forest model and preprocessor
rf_model = joblib.load("best_model.joblib")
preprocessor = joblib.load("preprocessor.joblib")

# Define a function to make predictions using the loaded Random Forest model
def predict_sales_revenue(month, customer_age, customer_gender, country, state, product_category, sub_category, quantity, unit_cost, unit_price):
    input_data = [[month, customer_age, customer_gender, country, state, product_category, sub_category, quantity, unit_cost, unit_price]]
    input_df = pd.DataFrame(input_data, columns=['Month', 'Customer Age', 'Customer Gender', 'Country', 'State', 'Product Category', 'Sub Category', 'Quantity', 'Unit Cost', 'Unit Price'])
    rf_pred = rf_model.predict(input_df)[0]
    # X_test_preprocessed = preprocessor.transform(input_df)
    # rf_pred = rf_model.predict(X_test_preprocessed)[0]
    return rf_pred

countries = ['United States', 'France', 'United Kingdom', 'Germany']
product_categories = ['Accessories', 'Clothing', 'Bikes']
sub_category = ['Tires and Tubes', 'Gloves', 'Helmets', 'Bike Stands',
       'Mountain Bikes', 'Hydration Packs', 'Jerseys', 'Fenders',
       'Cleaners', 'Socks', 'Caps', 'Touring Bikes', 'Bottles and Cages',
       'Vests', 'Road Bikes', 'Bike Racks', 'Shorts']
month = ['February', 'March', 'April', 'June', 'July', 'August',
       'September', 'October', 'November', 'December', 'May', 'January']

state = ['Washington', 'California', 'Oregon', 'Essonne', 'Yveline',
       'England', 'Hessen', 'Hamburg', 'Seine Saint Denis', 'Saarland',
       'Nordrhein-Westfalen', 'Bayern', 'Seine (Paris)', 'Pas de Calais',
       'Moselle', 'Hauts de Seine', 'Nord', 'Seine et Marne', 'Loiret',
       'Charente-Maritime', 'Loir et Cher', 'Brandenburg', 'Alabama',
       "Val d'Oise", 'Val de Marne', 'Minnesota', 'Wyoming', 'Ohio',
       'Garonne (Haute)', 'Kentucky', 'Texas', 'Missouri', 'Somme',
       'New York', 'Florida', 'Illinois', 'South Carolina',
       'North Carolina', 'Georgia', 'Virginia', 'Mississippi', 'Montana',
       'Arizona', 'Massachusetts', 'Utah']

# Define the input interface for the Gradio app
inputs = [
    gr.inputs.Dropdown(choices=month, label="Month"),
    gr.inputs.Number(label="Customer Age"),
    gr.inputs.Dropdown(choices=["F", "M"], label="Customer Gender"),
    gr.inputs.Dropdown(choices=countries, label="Country"),
    gr.inputs.Dropdown(choices=state, label="State"),
    gr.inputs.Dropdown(choices=product_categories, label="Product Category"),
    gr.inputs.Dropdown(choices=sub_category, label="Sub Category"),
    gr.inputs.Number(label="Quantity"),
    gr.inputs.Number(label="Unit Cost"),
    gr.inputs.Number(label="Unit Price")
]

# Define the Gradio interface and the function to make predictions
app = gr.Interface(fn=predict_sales_revenue, inputs=inputs, outputs="text")

# Launch the Gradio app
app.launch(share=True)


