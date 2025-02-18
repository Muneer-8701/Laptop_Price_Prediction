from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

# Load the model and DataFrame
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__, static_folder='static')

@app.route('/')
def home():
    return render_template('index.html', brands=df['Company'].unique(),
                           types=df['TypeName'].unique(),
                           cpus=df['Cpu brand'].unique(),
                           gpus=df['Gpu brand'].unique(),
                           oss=df['os'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        company = request.form['company']
        type_name = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        ips = 1 if request.form['ips'] == 'Yes' else 0
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # Process resolution
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

        # Create the input query
        query = np.array([company, type_name, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, 12)

        # Predict the price
        predicted_price = int(np.exp(pipe.predict(query)[0]))

        return render_template('index.html', prediction=f"The predicted price is ₹{predicted_price}",
                               brands=df['Company'].unique(),
                               types=df['TypeName'].unique(),
                               cpus=df['Cpu brand'].unique(),
                               gpus=df['Gpu brand'].unique(),
                               oss=df['os'].unique())
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}",
                               brands=df['Company'].unique(),
                               types=df['TypeName'].unique(),
                               cpus=df['Cpu brand'].unique(),
                               gpus=df['Gpu brand'].unique(),
                               oss=df['os'].unique())


if __name__ == '__main__':
    app.run(debug=True)

