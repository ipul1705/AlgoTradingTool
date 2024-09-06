from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView

class ToDoListApp(App):
    def build(self):
        # Layout utama menggunakan BoxLayout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Input field untuk menambahkan tugas baru
        self.task_input = TextInput(hint_text="Enter a new task", size_hint_y=None, height=40)
        main_layout.add_widget(self.task_input)

        # Tombol untuk menambah tugas
        add_task_btn = Button(text="Add Task", size_hint_y=None, height=40)
        add_task_btn.bind(on_press=self.add_task)
        main_layout.add_widget(add_task_btn)

        # ScrollView untuk menampilkan daftar tugas
        self.task_list = BoxLayout(orientation='vertical', spacing=10, size_hint_y=None)
        self.task_list.bind(minimum_height=self.task_list.setter('height'))

        scroll_view = ScrollView(size_hint=(1, 1))
        scroll_view.add_widget(self.task_list)

        main_layout.add_widget(scroll_view)

        return main_layout

    def add_task(self, instance):
        task_text = self.task_input.text.strip()
        if task_text:
            task_label = Label(text=task_text, size_hint_y=None, height=40)
            remove_btn = Button(text="Remove", size_hint_y=None, height=40)
            remove_btn.bind(on_press=lambda x: self.remove_task(task_label, remove_btn))
            
            task_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            task_layout.add_widget(task_label)
            task_layout.add_widget(remove_btn)

            self.task_list.add_widget(task_layout)
            self.task_input.text = ""  # Clear input field

    def remove_task(self, task_label, remove_btn):
        # Hapus widget tugas dari layout
        self.task_list.remove_widget(task_label.parent)

if __name__ == "__main__":
    ToDoListApp().run()
