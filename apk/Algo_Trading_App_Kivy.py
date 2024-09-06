import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.garden.matplotlib import FigureCanvasKivyAgg

# Fungsi untuk menghitung Keltner Channel
def calculate_keltner_channel(data, ema_period=20, atr_period=10, multiplier=2):
    # Menghitung Exponential Moving Average (EMA)
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

    # Menghitung Average True Range (ATR)
    data['TR'] = pd.DataFrame({
        'High-Low': data['High'] - data['Low'],
        'High-Close': abs(data['High'] - data['Close'].shift(1)),
        'Low-Close': abs(data['Low'] - data['Close'].shift(1))
    }).max(axis=1)
    
    data['ATR'] = data['TR'].rolling(window=atr_period).mean()

    # Menghitung Upper dan Lower Band
    data['Upper Band'] = data['EMA'] + multiplier * data['ATR']
    data['Lower Band'] = data['EMA'] - multiplier * data['ATR']
    
    return data

# Fungsi untuk menampilkan grafik Keltner Channel
def plot_keltner_channel(data, stock_symbol):
    plt.figure(figsize=(10, 6))
    plt.plot(data['Close'], label=f'{stock_symbol} Close', color='blue')
    plt.plot(data['EMA'], label='EMA', color='green')
    plt.plot(data['Upper Band'], label='Upper Band', color='red')
    plt.plot(data['Lower Band'], label='Lower Band', color='orange')
    plt.fill_between(data.index, data['Lower Band'], data['Upper Band'], color='lightgray', alpha=0.5)
    plt.title(f"Keltner Channel for {stock_symbol}")
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(loc='best')
    return plt.gcf()

# Kivy GUI
class AlgoTradingScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Input untuk simbol saham
        self.stock_input = TextInput(hint_text='Enter stock symbol (e.g. AAPL)', size_hint=(1, 0.1))
        layout.add_widget(self.stock_input)

        # Input untuk periode EMA
        self.ema_input = TextInput(hint_text='Enter EMA period (default 20)', size_hint=(1, 0.1))
        layout.add_widget(self.ema_input)

        # Input untuk periode ATR
        self.atr_input = TextInput(hint_text='Enter ATR period (default 10)', size_hint=(1, 0.1))
        layout.add_widget(self.atr_input)

        # Input untuk multiplier
        self.multiplier_input = TextInput(hint_text='Enter multiplier (default 2)', size_hint=(1, 0.1))
        layout.add_widget(self.multiplier_input)

        # Tombol untuk menampilkan Keltner Channel
        self.button = Button(text='Show Keltner Channel', size_hint=(1, 0.2))
        self.button.bind(on_press=self.on_button_press)
        layout.add_widget(self.button)

        # Area untuk grafik
        self.graph_area = BoxLayout(size_hint=(1, 0.7))
        layout.add_widget(self.graph_area)

        self.add_widget(layout)

    def on_button_press(self, instance):
        # Mengambil input pengguna
        stock_symbol = self.stock_input.text or 'AAPL'
        ema_period = int(self.ema_input.text) if self.ema_input.text else 20
        atr_period = int(self.atr_input.text) if self.atr_input.text else 10
        multiplier = float(self.multiplier_input.text) if self.multiplier_input.text else 2

        # Mengambil data saham dari Yahoo Finance
        stock_data = yf.download(stock_symbol, period="1y", interval="1d")

        # Menghitung Keltner Channel
        keltner_data = calculate_keltner_channel(stock_data, ema_period, atr_period, multiplier)

        # Menghapus grafik lama
        self.graph_area.clear_widgets()

        # Membuat grafik baru
        fig = plot_keltner_channel(keltner_data, stock_symbol)
        self.graph_area.add_widget(FigureCanvasKivyAgg(fig))

class AlgoTradingApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(AlgoTradingScreen(name='algo_trading'))
        return sm

if __name__ == '__main__':
    AlgoTradingApp().run()
