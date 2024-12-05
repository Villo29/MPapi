from flask import Flask, jsonify
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


app = Flask(__name__)

file_path = 'last_monts_ride.csv'
df = pd.read_csv(file_path)

df['create_at'] = pd.to_datetime(df['create_at'])
df['create_at_date'] = df['create_at'].dt.date
daily_counts = df.groupby('create_at_date').size()
daily_counts = daily_counts.asfreq('D', fill_value=0)

# Ajustar el modelo SARIMA
model = SARIMAX(daily_counts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
sarima_model = model.fit(disp=False)

# Generar predicciones
forecast = sarima_model.get_forecast(steps=30)
forecast_index = pd.date_range(daily_counts.index[-1] + pd.Timedelta(days=1), periods=30)
forecast_values = forecast.predicted_mean

# Convertir a un DataFrame para estructurar los resultados
forecast_df = pd.DataFrame({
    'date': forecast_index,
    'predicted_count': forecast_values
}).reset_index(drop=True)

# Ruta para obtener datos históricos
@app.route('/historical', methods=['GET'])
def get_historical_data():
    historical_data = daily_counts.reset_index()
    historical_data.columns = ['date', 'count']
    return jsonify(historical_data.to_dict(orient='records'))

# Ruta para obtener predicciones
@app.route('/forecast', methods=['GET'])
def get_forecast_data():
    return jsonify(forecast_df.to_dict(orient='records'))

# Ejecutar la aplicación
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
