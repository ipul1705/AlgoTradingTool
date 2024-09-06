from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen

class ColorfulScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # Label dengan warna latar belakang
        self.label = Label(text="Welcome to the Colorful App!",
                           font_size='24sp',
                           color=(1, 1, 1, 1),
                           size_hint=(1, 0.2),
                           bold=True)
        layout.add_widget(self.label)

        # TextInput dengan warna latar belakang
        self.text_input = TextInput(hint_text="Enter something colorful...",
                                    background_color=(0.7, 0.2, 0.4, 1),
                                    foreground_color=(1, 1, 1, 1),
                                    size_hint=(1, 0.2))
        layout.add_widget(self.text_input)

        # Tombol dengan warna latar belakang berbeda
        self.button = Button(text="Submit",
                             background_color=(0.2, 0.6, 0.4, 1),
                             color=(1, 1, 1, 1),
                             size_hint=(1, 0.2))
        self.button.bind(on_press=self.on_button_press)
        layout.add_widget(self.button)

        # Menambahkan layout ke screen
        self.add_widget(layout)

    def on_button_press(self, instance):
        input_text = self.text_input.text
        self.label.text = f"You entered: {input_text}"
        self.label.color = (0.2, 0.6, 0.9, 1)  # Mengubah warna teks

class ColorfulApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(ColorfulScreen(name="colorful"))
        return sm

if __name__ == '__main__':
    ColorfulApp().run()
