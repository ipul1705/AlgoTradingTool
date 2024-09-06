from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.progressbar import ProgressBar
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.spinner import Spinner


class AlgoTradingToolApp(App):
    def build(self):
        # Layout utama menggunakan BoxLayout
        main_layout = BoxLayout(orientation="horizontal")

        # Sidebar untuk disclaimer
        sidebar_layout = BoxLayout(orientation="vertical", size_hint=(0.3, 1), padding=10, spacing=10)
        disclaimer_text = (
            "Disclaimer\n\n"
            "1. No Financial Advice:\n"
            "   The information and tools\n"
            "   provided by Algo Trading Tool are\n"
            "   for stock trading reference only.\n\n"
            "2. Risk of Loss:\n"
            "   Stock Trading securities involves\n"
            "   substantial risk, including\n"
            "   the risk of loss...\n"
        )

        disclaimer_label = Label(text=disclaimer_text, halign="left", valign="top", color=(0, 1, 0, 1))
        sidebar_layout.add_widget(disclaimer_label)

        # Copyright
        copyright_label = Label(text="(c) 2024, saifurrohman1705@gmail.com", font_size=10, color=(1, 1, 1, 1))
        sidebar_layout.add_widget(copyright_label)

        # Area utama untuk input pengguna
        main_area_layout = BoxLayout(orientation="vertical", padding=10, spacing=10)

        # Label untuk judul
        title_label = Label(text="Algo Trading Tool", font_size=24, color=(0, 1, 0, 1))
        main_area_layout.add_widget(title_label)

        # Input untuk Stock Code
        stock_code_label = Label(text="Enter Stock Code:")
        stock_code_input = TextInput(multiline=False)
        main_area_layout.add_widget(stock_code_label)
        main_area_layout.add_widget(stock_code_input)

        # Combobox/Spinner untuk memilih tahun
        year_label = Label(text="Analyzed starting from what year?")
        year_spinner = Spinner(
            text="Select Year",
            values=("2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017"),
            size_hint=(None, None),
            size=(200, 44)
        )
        main_area_layout.add_widget(year_label)
        main_area_layout.add_widget(year_spinner)

        # Input untuk Periode dan ATR
        period_label = Label(text="Enter Period (days):")
        period_input = TextInput(text="20", multiline=False)
        main_area_layout.add_widget(period_label)
        main_area_layout.add_widget(period_input)

        atr_label = Label(text="Enter ATR (Average True Range):")
        atr_input = TextInput(text="10", multiline=False)
        main_area_layout.add_widget(atr_label)
        main_area_layout.add_widget(atr_input)

        # Tombol Prediksi dan Reset
        predict_button = Button(text="PREDICTION", size_hint=(0.5, None), height=44)
        reset_button = Button(text="RESET", size_hint=(0.5, None), height=44)
        button_layout = BoxLayout(size_hint=(1, None), height=50)
        button_layout.add_widget(predict_button)
        button_layout.add_widget(reset_button)
        main_area_layout.add_widget(button_layout)

        # Progress bar
        progress_label = Label(text="Progress:")
        progress_bar = ProgressBar(max=100)
        main_area_layout.add_widget(progress_label)
        main_area_layout.add_widget(progress_bar)

        # Tambahkan sidebar dan area utama ke layout utama
        main_layout.add_widget(sidebar_layout)
        main_layout.add_widget(main_area_layout)

        return main_layout


if __name__ == '__main__':
    AlgoTradingToolApp().run()

