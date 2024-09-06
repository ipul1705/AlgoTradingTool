from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout

class SimpleApp(App):
    def build(self):
        self.label = Label(text="Hello, World!")

        # Buat tombol dengan teks "Click Me"
        self.button = Button(text="Click Me")

        # Tambahkan event handler untuk tombol
        self.button.bind(on_press=self.on_button_press)

        # Atur layout dengan BoxLayout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.label)
        layout.add_widget(self.button)

        return layout

    def on_button_press(self, instance):
        # Ubah teks label saat tombol ditekan
        self.label.text = "Button Pressed!"

if __name__ == "__main__":
    SimpleApp().run()
