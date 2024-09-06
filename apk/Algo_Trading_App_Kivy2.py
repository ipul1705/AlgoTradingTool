import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.splitter import Splitter
from kivy.uix.image import Image
from io import BytesIO
from kivy.core.image import Image as CoreImage

# Function for Keltner Channel calculation
def calculate_keltner_channel(data, period=20):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    moving_average = typical_price.rolling(window=period).mean()
    true_range = data['High'] - data['Low']
    average_true_range = true_range.rolling(window=period).mean()
    
    upper_band = moving_average + 2 * average_true_range
    lower_band = moving_average - 2 * average_true_range
    
    return upper_band, lower_band

# Function for generating the chart
def generate_keltner_chart(data, upper_band, lower_band):
    plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close Price', color='blue')
    plt.plot(upper_band, label='Upper Band', color='green')
    plt.plot(lower_band, label='Lower Band', color='red')
    plt.fill_between(data.index, lower_band, upper_band, color='gray', alpha=0.1)
    plt.title("Keltner Channel Analysis")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return CoreImage(buf, ext='png').texture

# Function for creating a simple TensorFlow model for prediction
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function for training and predicting
def train_model(data):
    model = create_model()
    x = np.array(range(len(data)))  # Time-based index
    y = data['Close'].values  # Close prices

    model.fit(x, y, epochs=10, verbose=0)
    future_x = np.array([len(data) + i for i in range(5)])  # Predicting the next 5 days
    predictions = model.predict(future_x)
    return predictions

# Main Kivy Screen
class TradingScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        layout = BoxLayout(orientation='horizontal')

        # Left Sidebar for Keltner Channel
        keltner_sidebar = BoxLayout(orientation='vertical', size_hint=(0.4, 1))
        keltner_sidebar.add_widget(Label(text="Keltner Channel Analysis", font_size='20sp', bold=True))
        
        self.stock_input = TextInput(hint_text="Enter Stock Symbol", size_hint=(1, 0.1))
        keltner_sidebar.add_widget(self.stock_input)

        self.keltner_button = Button(text="Analyze", size_hint=(1, 0.1))
        self.keltner_button.bind(on_press=self.on_keltner_button_press)
        keltner_sidebar.add_widget(self.keltner_button)

        self.keltner_image = Image()
        keltner_sidebar.add_widget(self.keltner_image)

        # Right Sidebar for Machine Learning Predictions
        ml_sidebar = BoxLayout(orientation='vertical', size_hint=(0.4, 1))
        ml_sidebar.add_widget(Label(text="ML Price Prediction", font_size='20sp', bold=True))

        self.prediction_label = Label(text="Predicted Prices will appear here")
        ml_sidebar.add_widget(self.prediction_label)

        self.prediction_button = Button(text="Predict", size_hint=(1, 0.1))
        self.prediction_button.bind(on_press=self.on_prediction_button_press)
        ml_sidebar.add_widget(self.prediction_button)

        # Adding split layout
        layout.add_widget(keltner_sidebar)
        layout.add_widget(ml_sidebar)

        self.add_widget(layout)

    def on_keltner_button_press(self, instance):
        stock_symbol = self.stock_input.text.strip().upper()
        if stock_symbol:
            # Simulate fetching stock data (replace this with real stock data)
            dates = pd.date_range('2024-01-01', periods=100)
            data = pd.DataFrame({
                'Date': dates,
                'High': np.random.rand(100) * 100 + 50,
                'Low': np.random.rand(100) * 100 + 20,
                'Close': np.random.rand(100) * 100 + 40
            }).set_index('Date')

            upper_band, lower_band = calculate_keltner_channel(data)
            texture = generate_keltner_chart(data, upper_band, lower_band)
            self.keltner_image.texture = texture

    def on_prediction_button_press(self, instance):
        # Simulate fetching stock data (replace this with real stock data)
        dates = pd.date_range('2024-01-01', periods=100)
        data = pd.DataFrame({
            'Date': dates,
            'Close': np.random.rand(100) * 100 + 40
        }).set_index('Date')

        predictions = train_model(data)
        self.prediction_label.text = f"Predicted Prices: {predictions.flatten()}"

# Main Kivy App
class TradingApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(TradingScreen(name='trading'))
        return sm

if __name__ == '__main__':
    TradingApp().run()
