import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
import numpy as np
##import datetime
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from ta.volatility import KeltnerChannel
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.preprocessing import MinMaxScaler
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
import time
import webbrowser
import requests
import customtkinter
import ctypes

pd.options.mode.chained_assignment = None  # default='warn'
customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

global status_progress

##--> Langkah 1: Mengumpulkan Data Historis saham <--##
###-------------------------------------------------###
def fetch_data(saham, tahun):

    status_progress = "Collecting historical stock data"
    ##print(status_progress)

    start_date = tahun + '-01-01'

    # Mendapatkan tahun saat ini
    now = datetime.now()
    current_year = now.year
    end_year = str(current_year) + '-12-31'

    data = yf.download(saham, start=start_date, end=end_year)


    ##print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(100))

    df1 = pd.DataFrame(data)    

    update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value)

    return df1

##--> Langkah 2 : Menghitung Keltner Channel <--##
###--------------------------------------------###
def calculate_keltner_channel(df, periode_days, periode_atr):

    status_progress = "Calculating Keltner Channel Analysis"
    ##print(status_progress)

    ##print(f"Menghitung Keltner Channel dengan periode : {periode_days} dan periode ATR : {periode_atr}")
    keltner = KeltnerChannel(df['High'], df['Low'], df['Close'], window=periode_days, window_atr=periode_atr, fillna=True)
    df['Lower_Channel'] = keltner.keltner_channel_lband()
    df['Middle_Channel'] = keltner.keltner_channel_mband()
    df['Upper_Channel'] = keltner.keltner_channel_hband()

    ##print(df[['Close', 'Lower_Channel', 'Middle_Channel', 'Upper_Channel']].tail(100))

    update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value)

    return df


##--> Langkah 3 : Menentukan Sinyal Trading <--##
###-------------------------------------------###
def define_sinyal(data):
    data['Sell_Signal'] = np.where(data['Close'] < data['Lower_Channel'], 1, 0) # Sell Signal
    data['Buy_Signal'] = np.where(data['Close'] > data['Upper_Channel'], 1, 0)  # Buy Signal

    return data

def on_close():
    pass

##--> Langkah 4 : Memprediksi harga menggunakan model LSTM <--##
###----------------------------------------------------------###
def predict_stock(saham, tahun, periode_days, periode_atr):
    status_progress = "Predicting stock close price using Machine Learning"
    ##print(status_progress)

    data = fetch_data(saham, tahun)
    tahun = str(tahun)
    if data.empty or tahun == '' :
        messagebox.showerror("Error", "No data found for the stock symbol.")
        return

    data = calculate_keltner_channel(data, periode_days, periode_atr)
    define_sinyal(data)

    # Plot Keltner Channel
    plt.figure(figsize=(10,5))
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['Upper_Channel'], label='Upper Channel')
    plt.plot(data['Middle_Channel'], label='Middle Channel')
    plt.plot(data['Lower_Channel'], label='Lower Channel')
    plt.scatter(data[data['Buy_Signal'] == 1].index, data[data['Buy_Signal'] == 1]['Close'], label='Buy Signal', marker='^', color='green')
    plt.scatter(data[data['Sell_Signal'] == 1].index, data[data['Sell_Signal'] == 1]['Close'], label='Sell Signal', marker='v', color='red')
    plt.title(f'{saham} Price with Keltner Channel')
    plt.legend()
    plt.grid(True)

    # Prepare data for LSTM model
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna()

    ## Memproses Data untuk Model LSTM ##
    ##-------------------------------- ##
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    train_data_len = int(np.ceil(len(scaled_data) * .95))

    train_data = scaled_data[0:int(train_data_len), :]

    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    ## Membangun dan melatih model LSTM ##
    ##--------------------------------- ##
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    # Melatih model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=1)

    ## Memprediksi harga menggunakan model LSTM ##
    ##------------------------------------------##
    test_data = scaled_data[train_data_len - 60:, :]
    x_test = []
    y_test = data['Close'][train_data_len:].values

    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    global predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    valid = data[len(data) - len(predictions):]
    valid['Predictions'] = predictions

    # Plot prediksi harga
    plt.figure(figsize=(10,5))
    plt.plot(data['Close'][:train_data_len], label='Train Close Price')
    plt.plot(valid['Close'], label='Actual Close Price')
    plt.plot(valid['Predictions'], label='Predicted Close Price')
    ##plt.title(f'Stock Price Prediction for {saham}')
    plt.legend()
    plt.grid(True)


    valid = data[train_data_len:]
    valid['Predictions'] = predictions

    # Menentukan sinyal beli/jual berdasarkan prediksi harga
    valid['Future_Close'] = valid['Predictions'].shift(-1)
    valid['Buy_Recommendation'] = np.where((valid['Close'] < valid['Lower_Channel']) & (valid['Close'] < valid['Future_Close']), 1, 0)
    valid['Sell_Recommendation'] = np.where((valid['Close'] > valid['Upper_Channel']) & (valid['Close'] > valid['Future_Close']), 1, 0)

    # Plot rekomendasi
    ##plt.figure(figsize=(10,5))
    ##plt.plot(data['Close'], label='Close Price')
    ##plt.scatter(valid[valid['Buy_Recommendation'] == 1].index, valid[valid['Buy_Recommendation'] == 1]['Close'], label='Buy Recommendation', marker='^', color='green')
    ##plt.scatter(valid[valid['Sell_Recommendation'] == 1].index, valid[valid['Sell_Recommendation'] == 1]['Close'], label='Sell Recommendation', marker='v', color='red')
    ##plt.title(f'{saham} Price with Buy/Sell Recommendations')
    ##plt.legend()
    ##plt.grid(True)
    global plot_window
    plot_window = tk.Toplevel(main_area)
    plot_window.geometry("520x200+285+460")
    plot_window.title(f'3. Stock Price Prediction for {saham}')

    # Menonaktifkan tombol "x" (close)
    plot_window.protocol("WM_DELETE_WINDOW", on_close)

    # Membuat jendela selalu berada di atas
    plot_window.attributes("-topmost", True)

    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_window)
    canvas.draw()

    # Menambahkan toolbar Matplotlib (opsional)
    toolbar = NavigationToolbar2Tk(canvas, plot_window)
    toolbar.update()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=False)

    update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value)

##--> Langkah 5: Memberikan Rekomendasi Berdasarkan Hasil Analisis saham <--##
###------------------------------------------------------------------------###
def give_recommendation(saham, df2, predictions):
    status_progress = "Providing Recommendations Based on Stock Analysis and prediction close price Results"
    ##print(status_progress)

    last_close = df2['Close'].iloc[-1]
    next_predicted_close = float(predictions[-1])

    #Pengkondisian untuk recomendasi saham
    if  last_close > df2['Upper_Channel'].iloc[-1] and last_close > next_predicted_close :
        rekomendasi = 'Sell'
        because_rekomendasi = 'Because Close Price is bigger than Prediction Price and Upper Channel'
        bg = "green"
        
    elif last_close < df2['Lower_Channel'].iloc[-1] and last_close < next_predicted_close :
        rekomendasi = 'Buy'
        because_rekomendasi = 'Because Close Price is smaller than Prediction Price and Lower Channel'
        bg = "red"

    elif last_close > df2['Lower_Channel'].iloc[-1] and last_close < df2['Middle_Channel'].iloc[-1] and last_close > next_predicted_close :
        rekomendasi = 'Hold (prices tend to downtrend)'
        because_rekomendasi = ("Because Close Price is bigger than Prediction Price and Lower Channel\n" 
                                "but smaller than Middle Channel.")
        bg = "purple"

    elif last_close < df2['Lower_Channel'].iloc[-1] and last_close > next_predicted_close :
        rekomendasi = 'Hold (prices tend to downtrend)'
        because_rekomendasi = ("Because Close Price is bigger than Prediction Price but smaller\n" 
                               "than Lower Channel.")
        bg = "purple"

    elif last_close > df2['Middle_Channel'].iloc[-1] and last_close < df2['Upper_Channel'].iloc[-1] and last_close > next_predicted_close :
        rekomendasi = 'Hold (prices tend to uptrend)'
        because_rekomendasi = ("Because Close Price is bigger than Prediction Price and Middle Channel\n" 
                                "but smaller than Upper Channel.")
        bg = "orange"

    elif last_close > df2['Upper_Channel'].iloc[-1] and last_close < next_predicted_close :
        rekomendasi = 'Hold (prices tend to uptrend)'
        because_rekomendasi = ("Because Close Price is smaller than Prediction Price\n" 
                               "but bigger than Upper Channel.")    
        bg = "orange"

    elif last_close > df2['Lower_Channel'].iloc[-1] and last_close < df2['Middle_Channel'].iloc[-1] and last_close < next_predicted_close :
        rekomendasi = 'Hold (prices tend to downtrend)'
        because_rekomendasi = ("Because Close Price is smaller than Prediction Price and Middle Channel\n"
                               "but bigger than Lower Channel.")
        bg = "purple"

    elif last_close > df2['Lower_Channel'].iloc[-1] and last_close > df2['Middle_Channel'].iloc[-1] and last_close < df2['Upper_Channel'].iloc[-1] and last_close < next_predicted_close :
        rekomendasi = 'Hold (prices tend to uptrend)'
        because_rekomendasi = ("Because Close Price is smaller than Prediction Price and Upper Channel\n" 
                                "but bigger than Lower Channel and Middle Channel.")
        bg = "orange"



    ##else:
    ##    rekomendasi = 'Hold'
    ##    because_rekomendasi = 'Because Close Price is smaller than Prediction Price but bigger than Lower Channel'
    ##    bg = "orange"
        ##because_rekomendasi = rekomendasi1 + rekomendasi2 + rekomendasi3 + rekomendasi4 + rekomendasi5

    last_close = f"{last_close:.2f}"
    next_predicted_close = f"{next_predicted_close:.2f}"

    Lower_Channel = "[Lower Channel]  : " + f"{df2['Lower_Channel'].iloc[-1]:.2f}" 
    Middle_Channel ="[Middle Channel] : " + f"{df2['Middle_Channel'].iloc[-1]:.2f}"
    Upper_Channel = "[Upper Channel]  : " + f"{df2['Upper_Channel'].iloc[-1]:.2f}" 

    ##print(Lower_Channel)
    ##print(Middle_Channel)
    ##print(Upper_Channel)
    ##print(" ")

    harga_terakhir = "Closed Price   : " + str(last_close)
    harga_prediksi = "Predicted Price : " + str(next_predicted_close)
        
    ##print(harga_terakhir)
    ##print(harga_prediksi)
    ##print(" ")

            
    saham = str(saham)  
    rekomendasi = str(rekomendasi) 
    because_rekomendasi = str(because_rekomendasi)

    actual_close = "stock " + saham + ", actual Close Price today is : " + str(last_close)
    currenly_recomendation = "Currently recommendation for stock " + saham +" is : " + rekomendasi

    ##print(actual_close)
    ##print(currenly_recomendation)
    ##print(because_rekomendasi)

    global shadow_frame, main_frame, label_lower, label_middle, label_upper
    global label_kosong1, label_kosong2
    global label_closed, label_predicted
    global label_actual, label_currenly, label_because

    # Membuat frame untuk shadow
    shadow_frame = tk.Frame(main_area, width=520, height=300, bg='black')
    shadow_frame.pack_propagate(0)  # Mencegah frame mengubah ukurannya berdasarkan kontennya
    shadow_frame.place(x=7, y=145)

    # Membuat frame utama
    main_frame = tk.Frame(main_area, width=520, height=300, bg=bg)
    main_frame.pack_propagate(0)  # Mencegah frame mengubah ukurannya berdasarkan kontennya
    main_frame.place(x=3, y=140)  # Tempatkan frame utama sedikit bergeser dari shadow untuk membuat efek bayangan

    # Judul
    label_judul = tk.Label(main_frame, text=f"4. Recomendation for {saham}", font=("Helvetica", 8, "bold"), bg=bg, fg="white")
    label_judul.grid(row=0, column=0, padx=1, pady=1, sticky="W")

    # Lower Channel
    label_lower = tk.Label(main_frame, text=Lower_Channel, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_lower.grid(row=1, column=0, padx=1, pady=1, sticky="W")

    # Middle Channel
    label_middle = tk.Label(main_frame, text=Middle_Channel, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_middle.grid(row=2, column=0, padx=1, pady=1, sticky="W")


    # Upper Channel
    label_upper = tk.Label(main_frame, text=Upper_Channel, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_upper.grid(row=3, column=0, padx=1, pady=1, sticky="W")

    # spasi1
    label_kosong1 = tk.Label(main_frame, text="", font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_kosong1.grid(row=4, column=0, padx=1, pady=1, sticky="W")

    # closed price
    label_closed = tk.Label(main_frame, text=harga_terakhir, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_closed.grid(row=5, column=0, padx=1, pady=1, sticky="W")

    # predicted price
    label_predicted = tk.Label(main_frame, text=harga_prediksi, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_predicted.grid(row=6, column=0, padx=1, pady=1, sticky="W")

    # spasi2
    label_kosong2 = tk.Label(main_frame, text="", font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_kosong2.grid(row=7, column=0, padx=1, pady=1, sticky="W")

    # actual close
    label_actual = tk.Label(main_frame, text=actual_close, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_actual.grid(row=8, column=0, padx=1, pady=1, sticky="W")

    # currenly recomendation
    label_currenly = tk.Label(main_frame, text=currenly_recomendation, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_currenly.grid(row=9, column=0, padx=1, pady=1, sticky="W")

    # Label Because
    label_because = tk.Label(main_frame, text=because_rekomendasi, font=("Helvetica", 10, "bold"), bg=bg, fg="white")
    label_because.grid(row=10, column=0, padx=1, pady=1, sticky="W")

    ##-----SHADOW FRAME-------##
    # Judul
    label_judul = tk.Label(shadow_frame, text="4. Recomendation", font=("Helvetica", 8, "bold"), bg="black", fg="black")
    label_judul.grid(row=0, column=0, padx=1, pady=1, sticky="W")

    # Lower Channel
    label_lower = tk.Label(shadow_frame, text=Lower_Channel, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_lower.grid(row=1, column=0, padx=1, pady=1, sticky="W")

    # Middle Channel
    label_middle = tk.Label(shadow_frame, text=Middle_Channel, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_middle.grid(row=2, column=0, padx=1, pady=1, sticky="W")

    # Upper Channel
    label_upper = tk.Label(shadow_frame, text=Upper_Channel, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_upper.grid(row=3, column=0, padx=1, pady=1, sticky="W")

    # spasi1
    label_kosong1 = tk.Label(shadow_frame, text="", font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_kosong1.grid(row=4, column=0, padx=1, pady=1, sticky="W")

    # closed price
    label_closed = tk.Label(shadow_frame, text=harga_terakhir, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_closed.grid(row=5, column=0, padx=1, pady=1, sticky="W")

    # predicted price
    label_predicted = tk.Label(shadow_frame, text=harga_prediksi, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_predicted.grid(row=6, column=0, padx=1, pady=1, sticky="W")

    # spasi2
    label_kosong2 = tk.Label(shadow_frame, text="", font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_kosong2.grid(row=7, column=0, padx=1, pady=1, sticky="W")

    # actual close
    label_actual = tk.Label(shadow_frame, text=actual_close, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_actual.grid(row=8, column=0, padx=1, pady=1, sticky="W")

    # currenly recomendation
    label_currenly = tk.Label(shadow_frame, text=currenly_recomendation, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_currenly.grid(row=9, column=0, padx=1, pady=1, sticky="W")

    # Label Because for shadow frame
    label_because = tk.Label(shadow_frame, text=because_rekomendasi, font=("Helvetica", 10, "bold"), bg="black", fg="black")
    label_because.grid(row=10, column=0, padx=1, pady=1, sticky="W")

    update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value)

    status_progress = "Finish.                                                                                                "
    ##print(status_progress)

    update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value)

    root.config(cursor = "")
    root.update()

def update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value):
    root.config(cursor = "watch")
    root.update()

    for i in range(max_value + 1):
        time.sleep(0.05)  # Simulasi proses yang membutuhkan waktu
        progress_bar['value'] = i
        ##progress_label.config(text=f"Progress: {i}/{max_value}")
        progress_label.config(text=f"Progress : {status_progress}")
        root.update_idletasks()  # Memperbarui tampilan GUI

# Function to insert stock DataFrame history into Treeview
def insert_dataframe_history(tree_history_saham, df1):
    history_saham_label.place(x=535, y=150)
    frame_history_saham.place(x=535, y=170)
    tree_history_saham.pack(fill=tk.BOTH, expand=False)

    # Clear existing data in the treeview
    for row in tree_history_saham .get_children():
        tree_history_saham.delete(row)

    # Insert new data into treeview
    for index, row in df1.iterrows():
        tree_history_saham.insert("", "end", values=list(row))

# Function to insert keltner channel dataFrame into Treeview
def insert_keltner_channel(tree_keltner_channel, df2):
    keltner_channel_label.place(x=535, y=410)
    frame_keltner_channel.place(x=535, y=430)
    tree_keltner_channel.pack(fill=tk.BOTH, expand=False)

    # Clear existing data in the treeview
    for row in tree_keltner_channel.get_children():
        tree_keltner_channel.delete(row)

    # Insert new data into treeview
    for index, row in df2.iterrows():
        tree_keltner_channel.insert("", "end", values=list(row))

# Fungsi untuk validasi email  dan code
def validasi_prediksi(email_registration, code_registration, today_date_prediksi):
    for credential in user_credentials:
        
        #mendapatkan tanggal sekarang
        end_date_get = credential['end_date']
        format_tanggal_end_date = "%d/%m/%Y" 
           
        # Validasi format tanggal
        try:
            end_date = datetime.strptime(end_date_get, format_tanggal_end_date)
            ##print("Tanggal valid:", end_date)
    
        except ValueError as e:
            pass
        
        if credential['email'] == email_registration and credential['code'] == code_registration and end_date >= today_date_prediksi :
        ##if credential['email'] == email_registration and credential['code'] == code_registration and credential['end_date'] >= today_date_prediksi :
            result = {
                        "status" : "active"
                    }
            return result
        
        elif credential['email'] == email_registration and credential['code'] == code_registration and end_date < today_date_prediksi :
        ##elif credential['email'] == email_registration and credential['code'] == code_registration and credential['end_date'] < today_date_prediksi :
            result = {
                        "status" : "expired"
                    }
            return result
        
    result = {
                "status" : "non active"  
            }

    return result

##--> Langkah 5 : Klik Button Prediksi <--##
###--------------------------------------###
def on_predict(event):
    saham = saham_entry.get().upper()
    tahun = combobox_tahun.get()

    ##Variable Keltner Channel
    periode_days = int(periode_entry.get())  # Mengambil nilai dari entry kolom periode days
    periode_atr = int(atr_entry.get())  # Mengambil nilai dari entry kolom periode ATR
    ##print("periode Days : ",periode_days)
    ##print("periode ATR : ", periode_atr)

    #mendapatkan email, code dan tanggal sekarang
    email_registration = email_label.cget("text")
    code_registration = code_label.cget("text")

    hari_ini_prediksi = datetime.now()
    format_tanggal_prediksi_hari_ini = "%d/%m/%Y" 
    hari_ini_prediksi_str = hari_ini_prediksi.strftime(format_tanggal_prediksi_hari_ini)
    today_date_prediksi = datetime.strptime(hari_ini_prediksi_str, format_tanggal_prediksi_hari_ini)

    ##print("Email    : ",email_registration)
    ##print("Code     : ",code_registration)
    ##print("Hari ini : ",today_date_prediksi)

    if saham and tahun and periode_days and periode_atr :
        if check_internet_connection():
            ##print("Internet connection is available.")
            result = validasi_prediksi(email_registration, code_registration, today_date_prediksi)
            if result['status'] == 'expired':
                ##print("registration code is Expired!!!..")   
                messagebox.showerror("Error", "Expired registration code.\n"
                                        "Please contact saifurrohman1705@gmail.com")
                root.config(cursor="")
                root.update()
                saham_entry.focus_set()
        
            elif result['status'] == 'active':
                ##print("Registration code is valid.")
            
                fetch_data(saham, tahun)

                ##--> Data History Saham <--##
                df1 = fetch_data(saham, tahun)
                df1.reset_index(inplace=True)  # Reset index to use Date as a column
                insert_dataframe_history(tree_history_saham, df1)
                ##--------------------------##
        
                ##update_progress_bar_prediksi(progress_bar, progress_label, max_value)

                ##--> Menghitung Keltner Channel <--##
                data = fetch_data(saham, tahun)
                df2 = calculate_keltner_channel(data[['High', 'Low', 'Close']], periode_days, periode_atr)
                df2.reset_index(inplace=True)  # Reset index to use Date as a column
                insert_keltner_channel(tree_keltner_channel, df2)
                ##--------------------------##

                predict_stock(saham, tahun, periode_days, periode_atr)
                give_recommendation(saham, df2, predictions)

                ##update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value)

        else:
            ##print("No internet connection.")
            messagebox.showwarning("Error Message", "No internet connection.!!")
            root.config(cursor="")
            root.update()
            saham_entry.focus_set()
        
    else:
        messagebox.showwarning("Input Error", "Please enter a stock code, year, periode days and periode ATR!!!")
        root.config(cursor = "")
        root.update()
        saham_entry.focus_set()

##--> Langkah 5 : Klik Button Prediksi <--##
###--------------------------------------###
def on_predict_button_click():
    saham = saham_entry.get().upper()
    tahun = combobox_tahun.get()

    ##Variable Keltner Channel
    periode_days = int(periode_entry.get()) # Mengambil nilai dari entry periode days
    periode_atr = int(atr_entry.get())  # Mengambil nilai dari entry periode atr
    ##print("periode Days : ",periode_days)
    ##print("periode ATR : ", periode_atr)

    #mendapatkan email, code dan tanggal sekarang
    email_registration = email_label.cget("text")
    code_registration = code_label.cget("text")

    hari_ini_prediksi = datetime.now()
    format_tanggal_prediksi_hari_ini = "%d/%m/%Y" 
    hari_ini_prediksi_str = hari_ini_prediksi.strftime(format_tanggal_prediksi_hari_ini)
    
    # Format tanggal: Tahun-Bulan-Hari
    today_date_prediksi = datetime.strptime(hari_ini_prediksi_str, format_tanggal_prediksi_hari_ini)

    if saham and tahun and periode_days and periode_atr :
        if check_internet_connection():
            ##print("Internet connection is available.")
            result = validasi_prediksi(email_registration, code_registration, today_date_prediksi)
            if result['status'] == 'expired':
                ##print("registration code is Expired!!!..")   
                messagebox.showerror("Error", "Expired registration code.\n"
                                        "Please contact saifurrohman1705@gmail.com")
                root.config(cursor="")
                root.update()
                saham_entry.focus_set()
        
            elif result['status'] == 'active':
                ##print("Registration code is valid.")
            
                fetch_data(saham, tahun)

                ##--> Data History Saham <--##
                df1 = fetch_data(saham, tahun)
                df1.reset_index(inplace=True)  # Reset index to use Date as a column
                insert_dataframe_history(tree_history_saham, df1)
                ##--------------------------##
        
                ##update_progress_bar_prediksi(progress_bar, progress_label, status_progress, max_value)


                ##--> Menghitung Keltner Channel <--##
                data = fetch_data(saham, tahun)
                df2 = calculate_keltner_channel(data[['High', 'Low', 'Close']], periode_days, periode_atr)
                df2.reset_index(inplace=True)  # Reset index to use Date as a column
                insert_keltner_channel(tree_keltner_channel, df2)
                ##--------------------------##

                predict_stock(saham, tahun, periode_days, periode_atr)
                give_recommendation(saham, df2, predictions)

                
        else:
            ##print("No internet connection.")
            messagebox.showwarning("Error Message", "No internet connection.!!")
            root.config(cursor="")
            root.update()
            saham_entry.focus_set()
        
    else:
        messagebox.showwarning("Input Error", "Please enter a stock code, year, periode days and periode ATR!!!")
        root.config(cursor = "")
        root.update()
        saham_entry.focus_set()


def clear_entries():
    # Mengosongkan entry
    saham_entry.delete(0, tk.END)

    # Mengosongkan combobox
    combobox_tahun.set('')

    #mengembalikan mouse ke semula
    root.config(cursor = "")
    root.update()

    # Mengosongkan data di table history saham
    for row in tree_history_saham .get_children():
        tree_history_saham.delete(row)

    # Menyembunyikan label, treeview dan frame saham
    history_saham_label.place_forget()
    frame_history_saham.place_forget()  
    tree_history_saham.pack_forget()

    # Mengosongkan data di table history saham
    for row in tree_keltner_channel.get_children():
        tree_keltner_channel.delete(row)

    # Menyembunyikan label, treeview dan frame keltner channel
    keltner_channel_label.place_forget()
    frame_keltner_channel.place_forget()  
    tree_keltner_channel.pack_forget()

    periode_entry.delete(0, tk.END)  # Menghapus teks yang ada di periode Entry
    periode_entry.insert(0, "20")  

    atr_entry.delete(0, tk.END)  # Menghapus teks yang ada di ATR Entry
    atr_entry.insert(0, "10")  

    #hide frame and label recomendation
    shadow_frame.place_forget()
    main_frame.place_forget() 
    label_lower.pack_forget()
    label_middle.pack_forget()
    label_upper.pack_forget()
    label_closed.pack_forget()
    label_predicted.pack_forget()
    label_actual.pack_forget()
    label_currenly.pack_forget()
    label_because.pack_forget()


    # Mengembalikan value = 0 di progress bar
    progress_bar['value'] = 0
    progress_label.config(text="Progress : Start...")

    saham_entry.focus_set()

    #menyembunyikan window plot
    plot_window.destroy()


#sortir treeview history saham
def sort_treeview_saham(tree_history_saham, col, descending):
    # Mendapatkan semua item dari Treeview
    data = [(tree_history_saham.set(child, col), child) for child in tree_history_saham.get_children('')]

    # Mengurutkan data berdasarkan kolom
    data.sort(reverse=descending)

    # Menghapus semua item dari Treeview
    for index, (val, child) in enumerate(data):
        tree_history_saham.move(child, '', index)

    # Mengatur ulang heading untuk mencerminkan arah pengurutan
    tree_history_saham.heading(col, command=lambda: sort_treeview_saham(tree_history_saham, col, not descending))

#sortir treeview keltner channel
def sort_treeview_keltner(tree_keltner_channel, col, descending):
    # Mendapatkan semua item dari Treeview
    data = [(tree_keltner_channel.set(child, col), child) for child in tree_keltner_channel.get_children('')]

    # Mengurutkan data berdasarkan kolom
    data.sort(reverse=descending)

    # Menghapus semua item dari Treeview
    for index, (val, child) in enumerate(data):
        tree_keltner_channel.move(child, '', index)

    # Mengatur ulang heading untuk mencerminkan arah pengurutan
    tree_keltner_channel.heading(col, command=lambda: sort_treeview_keltner(tree_keltner_channel, col, not descending))    

# Membulatkan angka di table
def format_data_with_two_decimals(tree_keltner_channel):
    for child in tree_keltner_channel.get_children():
        # Mendapatkan nilai saat ini dari item
        values = tree_keltner_channel.item(child, 'values')
        # Memformat nilai agar memiliki dua angka di belakang koma
        formatted_values = [f"{float(value):.2f}" if value.replace('.', '', 11).isdigit() else value for value in values]
        # Memperbarui item dengan nilai yang telah diformat
        tree_keltner_channel.item(child, values=formatted_values)

def on_enter_saham(event):
    combobox_tahun.focus_set()

def on_enter_periode(event):
    periode_entry.focus_set()        

def on_enter_atr(event):
    atr_entry.focus_set()        

# Fungsi untuk menyaring item berdasarkan input dan memunculkan sugesti
def filter_items(event):
    value = event.widget.get().lower()
    ##value = combobox_tahun.get().lower()
    if value == '':
        combobox_tahun['values'] = suggestions
    else:
        filtered_items = [item for item in suggestions if value in item.lower()]
        combobox_tahun['values'] = filtered_items

    combobox_tahun.event_generate('<Down>')  # Memunculkan dropdown untuk menampilkan sugesti

def on_key_release(event):
    saham_entry_text = saham_entry.get().upper()
    saham_entry.delete(0, tk.END)
    saham_entry.insert(0, saham_entry_text)

def only_numbers(char):
    return char.isdigit()

def is_valid_char(char):
    return char.isalpha() or char == '.' or char == ''

def validate_input(P):
    return all(is_valid_char(c) for c in P)

def user_tutorial():
    # Gantilah tautan ini dengan tautan berbagi Google Drive Anda
    pdf_url = 'https://drive.google.com/file/d/1zio98znLs7u3bxQ6GeaYq_WEbHIB_f8_/view?usp=drive_link'
    webbrowser.open_new(pdf_url)
    ##messagebox.showinfo("User Manual", "Open file user manual")

def about():
    messagebox.showinfo("About", "(C)Copyright 2024, saifurrohman1705@gmail.com")    

def on_enter_registration(event):
    reg_code_entry.focus_set()

# Fungsi untuk validasi email  dan code
def validasi(email, code, today_date):
    registration_window.config(cursor="watch")
    registration_window.update()
    
    for credential in user_credentials:
        #mendapatkan tanggal sekarang
        end_date_get = credential['end_date']
        format_tanggal_end_date = "%d/%m/%Y" 
           
        # Validasi format tanggal
        try:
            end_date = datetime.strptime(end_date_get, format_tanggal_end_date)
            ##print("Tanggal valid:", end_date)
    
        except ValueError as e:
            pass
        
        ##if credential['email'] == email and credential['code'] == code and credential['end_date'] > today_date :
        if credential['email'] == email and credential['code'] == code and end_date >= today_date :    
            result = {
                        "status" : "active"
                    }
            return result

        ##elif credential['email'] == email and credential['code'] == code and credential['end_date'] < today_date :
        elif credential['email'] == email and credential['code'] == code and end_date < today_date :    
            result = {
                        "status" : "expired"
                    }
            return result
        
    result = {
                "status" : "non active"  
            }
    return result

def registration_validation():
    registration_window.config(cursor="watch")
    registration_window.update()
    ##--AMBIL DATA EMAIL DAN REGISTRATION CODE DARI GGOGLE DRIVE--##
    # ID file Google Drive
    file_id = '1UJt2sQ-TihcSuRHbB6eaEOiI0KO-p0bh'
    # URL unduh langsung Google Drive
    download_url = f'https://drive.google.com/uc?id={file_id}'

    # Membaca file CSV ke dalam DataFrame pandas
    df = pd.read_csv(download_url)

    # Mengambil data sebagai list of tuples
    data_tuples = [tuple(row) for row in df.to_records(index=False)]

    # Mengurai data menjadi tuples
    global user_credentials 
    user_credentials = []
    for item in data_tuples:
        email, code, start_date, duration, end_date = item[0].split(';')
        user_credentials.append({'email': email, 'code': code, 'start_date': start_date, 'duration':duration, 'end_date':end_date})

    # Cetak hasilnya
    ##for credential in user_credentials:
        ##print(f"Email: {credential['email']}, Code: {credential['code']}, end_date: {credential['end_date']}")
        ##print("End Date : ",credential['end_date'])

    global entered_email, entered_code
    entered_email = email_entry.get()
    entered_code = reg_code_entry.get()

    #mendapatkan tanggal sekarang
    hari_ini = datetime.now()
    format_tanggal_hari_ini = "%d/%m/%Y" 
    hari_ini_str = hari_ini.strftime(format_tanggal_hari_ini)
    
    # Format tanggal: Tahun-Bulan-Hari
    today_date = datetime.strptime(hari_ini_str, format_tanggal_hari_ini)
    ##print("Hari ini : ",today_date)

    if entered_email == '':
        messagebox.showerror("Error", "Please fill in your email.")
        registration_window.config(cursor="")
        registration_window.update()
        email_entry.focus_set()

    elif entered_code == '':
        messagebox.showerror("Error", "Please fill in the registration code.\n"
                                    "Please contact saifurrohman1705@gmail.com")    
        registration_window.config(cursor="")
        registration_window.update()
        reg_code_entry.focus_set()

    else :
        result = validasi(entered_email, entered_code, today_date)
        if result['status'] == 'active':
            ##print("Registration code is valid.")
            registration_window.config(cursor="")
            registration_window.update()
            registration_window.destroy()
            create_main_window()
            

        elif result['status'] == 'expired':
            ##print("registration code is Expired!!!..")
            messagebox.showerror("Error", "Expired registration code.\n"
                                          "Please contact saifurrohman1705@gmail.com")
            registration_window.config(cursor="")
            registration_window.update()
            reg_code_entry.focus_set()
            

        elif result['status'] == 'non active':
            ##print("Invalid registration code!!!.")
            messagebox.showerror("Error", "Invalid registration code.\n"
                                          "Please contact saifurrohman1705@gmail.com")
            registration_window.config(cursor="")
            registration_window.update()
            reg_code_entry.focus_set()
            


def registration_validation_event(event):
    registration_window.config(cursor="watch")
    registration_window.update()
    ##--AMBIL DATA EMAIL DAN REGISTRATION CODE DARI GGOGLE DRIVE--##
    # ID file Google Drive
    file_id = '1UJt2sQ-TihcSuRHbB6eaEOiI0KO-p0bh'

    # URL unduh langsung Google Drive
    download_url = f'https://drive.google.com/uc?id={file_id}'

    # Membaca file CSV ke dalam DataFrame pandas
    df = pd.read_csv(download_url)

    # Mengambil data sebagai list of tuples
    data_tuples = [tuple(row) for row in df.to_records(index=False)]

    # Mengurai data menjadi list
    global user_credentials, entered_email, entered_code 
    user_credentials = []
    for item in data_tuples:
        email, code, start_date, duration, end_date = item[0].split(';')
        user_credentials.append({'email': email, 'code': code, 'start_date': start_date, 'duration':duration, 'end_date':end_date})

    # Cetak hasilnya
    ##for credential in user_credentials:
    ##    print(f"Email: {credential['email']}, Code: {credential['code']}, end_date: {credential['end_date']}")


    entered_email = email_entry.get()
    entered_code = reg_code_entry.get()

    #mendapatkan tanggal sekarang
    hari_ini = datetime.now()
    format_tanggal_hari_ini = "%d/%m/%Y" 
    hari_ini_str = hari_ini.strftime(format_tanggal_hari_ini)
    ##today_date = hari_ini.strftime("%d/%m/%Y")
    

     # Format tanggal: Tahun-Bulan-Hari
    today_date = datetime.strptime(hari_ini_str, format_tanggal_hari_ini)
    ##print("Hari ini : ",today_date)

    if entered_email == '':
        messagebox.showerror("Error", "Please fill in your email.")
        registration_window.config(cursor="")
        registration_window.update()
        email_entry.focus_set()

    elif entered_code == '':
        messagebox.showerror("Error", "Please fill in the registration code.\n"
                                      "Please contact saifurrohman1705@gmail.com")   
        registration_window.config(cursor="")
        registration_window.update()
        reg_code_entry.focus_set()

    else :
        result = validasi(entered_email, entered_code, today_date)
        if result['status'] == 'active':
            ##print("Registration code is valid.")
            registration_window.config(cursor="")
            registration_window.update()
            registration_window.destroy()
            create_main_window()
            

        elif result['status'] == 'expired':
            ##print("registration code is Expired!!!..")   
            messagebox.showerror("Error", "Expired registration code.\n"
                                          "Please contact saifurrohman1705@gmail.com")
            registration_window.config(cursor="")
            registration_window.update()
            reg_code_entry.focus_set()
            
            
        elif result['status'] == 'non active':
            ##print("Invalid registration code!!!")
            messagebox.showerror("Error", "Invalid registration code.\n"
                                          "Please contact saifurrohman1705@gmail.com")
            registration_window.config(cursor="")
            registration_window.update()
            reg_code_entry.focus_set()
        
    

def create_registration_widgets():
    global registration_window, email_entry, reg_code_entry

    registration_window = tk.Tk()
    registration_window.title("Submit Form")
    registration_window.resizable(False, False)
    registration_window.iconbitmap("icon/analytics.ico")
    width = 440
    height = 200

    ##--Center Screen--##
    screen_width = registration_window.winfo_screenwidth()
    screen_height = registration_window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    registration_window.geometry(f'{width}x{height}+{x}+{y}')
    registration_window.configure(bg='#a9a9a9')  # Set background color


    # Menambahkan label di area utama
    note_label = tk.Label(registration_window, bg='#a9a9a9', text="Note: for registration code, please contact saifurrohman1705@gmail.com", font=('arial', 10))
    note_label.place(x=3, y=6)

    email_label = tk.Label(registration_window, bg='#a9a9a9', text="Enter Email :")
    email_label.place(x=5, y=70)

    email_entry = tk.Entry(registration_window, width=30)
    email_entry.place(x=143, y=70)
    email_entry.focus_set() 
    email_entry.bind("<Return>", on_enter_registration)

    reg_code_label = tk.Label(registration_window, bg='#a9a9a9', text="Enter Registration Code :")
    reg_code_label.place(x=5, y=100)

    reg_code_entry = tk.Entry(registration_window, width=30, show='#')
    reg_code_entry.place(x=143, y=100)
    reg_code_entry.bind("<Return>", registration_validation_event)

    button_submit = ttk.Button(registration_window, text="Submit", style= "TButton", width=29, command=registration_validation)
    button_submit.place(x=143, y=125)

    registration_window.mainloop()

# Fungsi untuk membuat tampilan utama
def create_main_window():

    # Membuat jendela utama
    global root
    root = tk.Tk()

    root.title("Algo Trading Tool")

    # Maximize the window
    root.state('zoomed')

    # Menonaktifkan tombol maximize
    root.resizable(False, False)

    # Mengatur ikon jendela menggunakan file .ico
    root.iconbitmap("icon/analytics.ico")
    # Mengatur ukuran jendela
    ##root.geometry("600x400")
    root.geometry(f"{800}x{600}")

    global main_area

    # Membuat frame utama
    main_area = tk.Frame(root, bg='#a9a9a9', width=500, height=600)
    main_area.pack(side='right', fill='both', expand=True)

    ##{--SIDEBAR--}##
    # Membuat frame sidebar
    sidebar = tk.Frame(root, width=250, bg='#2c3e50', height=600, relief='sunken', borderwidth=2)
    sidebar.pack(side='left', fill='y')

    sub_main_label = (
        "                Tool for stock analysis with Keltner Channel Analysis (default parameter : periode = 20 and ATR = 10)\n"
        "                                   and prediction close price with Machine Learning (Tensorflow Keras)\n"
    )

    disclaimer_text = (
        "Disclaimer\n\n"
        "1.No Financial Advice:\n" 
        "  The information and tools\n"
        "  provided by Algo Trading Tool are\n" 
        "  for stock trading reference only.\n\n"
        "2.Risk of Loss:\n"
        "  Stock Trading securities involves\n"
        "  substantial risk, including\n"
        "  the risk of loss. You should\n" 
        "  consider your financial situation\n" 
        "  and consult with a financial\n"
        "  advisor before making any\n"
        "  investment decisions.\n\n"
        "3.Performance Not Guaranteed:\n" 
        "  Use Algo Trading Tool is for\n"
        "  reference only does not\n" 
        "  guarantee any specific\n"
        "  results or profits.\n\n"
        "4.Liability Limitation:\n" 
        "  I am not responsible for\n"
        "  any losses arising from\n"
        "  the use of Algo Trading Tool\n\n"
        "5.User Responsibility:\n"
        "  Users are solely responsible\n"
        "  for their own trading decisions\n" 
        "  and outcomes. Algo Trading Tool\n"
        "  is only a tool to aid in trading\n"
        "  decisions and should not be\n"
        "  solely relied upon.\n\n"
    )

    ##, wrap=tk.WORD   
    text_widget = tk.Text(sidebar, bg='#2c3e50', fg="lightgreen", height=35, width=35, borderwidth=0, highlightthickness=0)
    text_widget.insert(tk.END, disclaimer_text)
    text_widget.pack(padx=0, pady=0)

    copyright_label = tk.Label(sidebar, bg='#2c3e50', fg="white", text="(c) copyright 2024, saifurrohman1705@gmail.com", font=('Helvetica', 8, 'bold'))
    copyright_label.place(x=3, y=656)

    # Menambahkan label di area utama
    main_label = tk.Label(main_area, bg='#a9a9a9', fg="green", text="Algo Trading Tool", font=('Helvetica', 20, 'bold'))
    main_label.place(x=350, y=5)

    sub_main_label = tk.Label(main_area, bg='#a9a9a9', fg="black", text=sub_main_label, font=('Arial', 8, 'bold'))
    sub_main_label.place(x=120, y=38)

    global email_label, code_label
    email_label = tk.Label(main_area, bg='#a9a9a9', fg='#a9a9a9', width=20, text=entered_email)
    email_label.place(x=5, y=10)

    code_label = tk.Label(main_area, bg='#a9a9a9', fg='#a9a9a9', width=20, text=entered_code)
    code_label.place(x=5, y=30)

    # Menambahkan label untuk saham
    label_saham = tk.Label(main_area, bg='#a9a9a9', text="Enter Stock Code                               :")
    label_saham.place(x=5, y=80)

    ##vcmd1 = (root.register(validate_input), '%P')

    # Menambahkan Entry
    global saham_entry
    saham_entry = ttk.Entry(main_area, width=33)##, validate='key', validatecommand=vcmd1)
    saham_entry.place(x=200, y=80)
    saham_entry.focus_set()  # Set focus to the entry widget
    saham_entry.bind("<Return>", on_enter_saham)
    saham_entry.bind("<KeyRelease>", on_key_release)

    # Menambahkan label untuk year
    label_tahun = tk.Label(main_area, bg='#a9a9a9', text="Analyzed starting from what year? :")
    label_tahun.place(x=5, y=110)

    vcmd2 = (main_area.register(only_numbers), '%S')

    # Menambahkan Combobox tahun
    global combobox_tahun, suggestions
    combobox_tahun = ttk.Combobox(main_area, width=30, validate='key', validatecommand=vcmd2)
    combobox_tahun.place(x=200, y=110)
    combobox_tahun.bind("<KeyRelease>", filter_items)
    combobox_tahun.bind("<Return>", on_enter_periode)

    # Daftar saran
    suggestions = ["2024", "2023", "2022", "2021" , "2020", "2019", "2018", "2017", "2016" , "2015", "2014", "2013", "2012", "2011", "2010" , "2009", "2008", "2007", "2006", "2005", "2004" , "2003", "2002", "2001" , "2000"]
    combobox_tahun['values'] = suggestions    
        
    # Menambahkan kolom periode days dan periode ATR
    global periode_entry, atr_entry
    periode_label = tk.Label(main_area, bg='#a9a9a9', text="Enter Periode (days)                          :")
    periode_label.place(x=5, y=140)

    periode_entry = tk.Entry(main_area, width=10)
    periode_entry.place(x=200, y=140)
    periode_entry.bind("<Return>", on_enter_atr) 
    periode_entry.insert(0, "20")  # 0 menunjukkan bahwa teks dimasukkan pada posisi pertama


    atr_label = tk.Label(main_area, bg='#a9a9a9', text="Enter ATR (Average True Range)     :")
    atr_label.place(x=5, y=170)

    atr_entry = tk.Entry(main_area, width=10)
    atr_entry.place(x=200, y=170)
    atr_entry.bind("<Return>", on_predict)
    atr_entry.insert(0, "10")  # 0 menunjukkan bahwa teks dimasukkan pada posisi pertama

    periode_label2 = tk.Label(main_area, bg='#a9a9a9', text="setting parameter 1 for Keltner Channel")
    periode_label2.place(x=260, y=140)

    atr_label2 = tk.Label(main_area, bg='#a9a9a9', text="setting parameter 2 for Keltner Channel")
    atr_label2.place(x=260, y=170)

    global progress_bar, progress_label, max_value

    max_value = 100
    status_progress = "Start..."

    progress_bar = ttk.Progressbar(main_area, orient="horizontal", length=1040, mode="determinate")
    progress_bar.place(x=1, y=660)
    progress_bar['maximum'] = max_value

    ##progress_label = tk.Label(main_area, bg='#a9a9a9', text="Progress: 0/100")
    progress_label = tk.Label(main_area, bg='#a9a9a9', fg="red", text=f"Progress : {status_progress}", font=("Helvetica", 8, "bold"))
    progress_label.place(x=1, y=645)

    # Membuat gaya kustom untuk tombol
    style = ttk.Style()
    style.theme_use('clam')  # Pilih tema dasar

    # Mengonfigurasi gaya untuk tombol kustom
    style.configure('TButton',
                font=('Helvetica', 10, 'bold'),
                foreground='white',
                background='#3498db',
                borderwidth=1,
                focuscolor=style.configure(".")["background"],
                padding=3)

    style.map('TButton',
        foreground=[('active', 'white'), ('disabled', '#d9d9d9')],
        background=[('active', '#2980b9'), ('disabled', '#f0f0f0')],
        relief=[('pressed', 'sunken'), ('!pressed', 'raised')])

    # Menambahkan tombol Prediksi
    button_prediksi = ttk.Button(main_area, text="PREDICTION", style= "TButton", width=25, command=on_predict_button_click)
    ##button_prediksi.place(x=5, y=150)
    button_prediksi.place(x=410, y=75)

    # Menambahkan tombol Clear
    button_clear = ttk.Button(main_area, text="RESET", style= "TButton", width=25, command=clear_entries)
    ##button_clear.place(x=185, y=150)
    button_clear.place(x=410, y=106)

    ##-->Menambahkan table data untuk history saham<--##
    global history_saham_label, tree_history_saham, frame_history_saham


    history_saham_label = tk.Label(main_area, bg='#a9a9a9', text="1. History Data Stock :", font=('Helvetica', 8, 'bold'))
    history_saham_label.place_forget()

    # Membuat frame untuk menempatkan tabel
    frame_history_saham = ttk.Frame(main_area, width=10, height=10)
    frame_history_saham.place_forget()  

    # Membuat scrollbar vertikal
    scrollbar_history_saham = ttk.Scrollbar(frame_history_saham, orient=tk.VERTICAL)
    scrollbar_history_saham.pack(side=tk.RIGHT, fill=tk.Y)

    # Membuat tabel
    tree_history_saham = ttk.Treeview(frame_history_saham, columns=("Date", "Open", "High", "Low", "Close", "Volume"), show='headings', yscrollcommand=scrollbar_history_saham.set)
    tree_history_saham.pack_forget()

    # Mengkonfigurasi scrollbar
    scrollbar_history_saham.config(command=tree_history_saham.yview)

    # Menambahkan heading kolom
    tree_history_saham.heading("Date", text="Date", command=lambda: sort_treeview_saham(tree_history_saham, "Date", False))
    tree_history_saham.heading("Open", text="Open", command=lambda: sort_treeview_saham(tree_history_saham, "Open", False) )
    tree_history_saham.heading("High", text="High", command=lambda: sort_treeview_saham(tree_history_saham, "High", False))
    tree_history_saham.heading("Low", text="Low", command=lambda: sort_treeview_saham(tree_history_saham, "Low", False))
    tree_history_saham.heading("Close", text="Close", command=lambda: sort_treeview_saham(tree_history_saham, "Close", False))
    tree_history_saham.heading("Volume", text="Volume", command=lambda: sort_treeview_saham(tree_history_saham, "Volume", False))

    # Mengatur ukuran kolom
    tree_history_saham.column("Date", width=68, anchor='w')
    tree_history_saham.column("Open", width=80, anchor='e')
    tree_history_saham.column("High", width=80, anchor='e')
    tree_history_saham.column("Low", width=80, anchor='e')
    tree_history_saham.column("Close", width=80, anchor='e')
    tree_history_saham.column("Volume", width=80, anchor='e')
    ##-----------------------------------------------------##

    ##-->Menambahkan table data untuk Keltner Channel<--##
    global keltner_channel_label, tree_keltner_channel, frame_keltner_channel


    keltner_channel_label = tk.Label(main_area, bg='#a9a9a9', text="2. Calculate Keltner Channel Analysis :", font=('Helvetica', 8, 'bold'))
    keltner_channel_label.place_forget()

    # Membuat frame untuk menempatkan tabel
    frame_keltner_channel = ttk.Frame(main_area, width=10, height=50)
    frame_keltner_channel.place_forget()  

    # Membuat scrollbar vertikal
    scrollbar_keltner_channel = ttk.Scrollbar(frame_keltner_channel, orient=tk.VERTICAL)
    scrollbar_keltner_channel.pack(side=tk.RIGHT, fill=tk.Y)

    # Membuat tabel
    tree_keltner_channel = ttk.Treeview(frame_keltner_channel, columns=("Date", "High", "Low", "Close", "Lower_Channel", "Middle_Channel", "Upper_Channel"), show='headings', yscrollcommand=scrollbar_keltner_channel.set)
    tree_keltner_channel.pack_forget()

    # Mengkonfigurasi scrollbar
    scrollbar_keltner_channel.config(command=tree_keltner_channel.yview)

    # Menambahkan heading kolom
    tree_keltner_channel.heading("Date", text="Date", command=lambda: sort_treeview_keltner(tree_keltner_channel, "Date", False))
    tree_keltner_channel.heading("High", text="High", command=lambda: sort_treeview_keltner(tree_keltner_channel, "High", False))
    tree_keltner_channel.heading("Low", text="Low", command=lambda: sort_treeview_keltner(tree_keltner_channel, "Low", False))
    tree_keltner_channel.heading("Close", text="Close", command=lambda: sort_treeview_keltner(tree_keltner_channel, "Close", False))
    tree_keltner_channel.heading("Lower_Channel", text="Lower Channel", command=lambda: sort_treeview_keltner(tree_keltner_channel, "Lower_Channel", False))
    tree_keltner_channel.heading("Middle_Channel", text="Middle Channel", command=lambda: sort_treeview_keltner(tree_keltner_channel, "Middle_Channel", False))
    tree_keltner_channel.heading("Upper_Channel", text="Upper Channel", command=lambda: sort_treeview_keltner(tree_keltner_channel, "Upper_Channel", False))

    # Mengatur ukuran kolom
    tree_keltner_channel.column("Date", width=68, anchor='w')
    tree_keltner_channel.column("High", width=0, stretch=tk.NO, anchor='e')
    tree_keltner_channel.column("Low", width=0, stretch=tk.NO, anchor='e')
    tree_keltner_channel.column("Close", width=100, anchor='e')
    tree_keltner_channel.column("Lower_Channel", width=100, anchor='e')
    tree_keltner_channel.column("Middle_Channel", width=100, anchor='e')
    tree_keltner_channel.column("Upper_Channel", width=100, anchor='e')

    # Memformat data di Treeview agar memiliki dua angka di belakang koma
    format_data_with_two_decimals(tree_keltner_channel)

    # Mengurutkan kolom "Date" secara descending saat aplikasi dimulai
    ##root.after(1000, lambda: sort_treeview(tree_history_saham, "Date", True))

    # Membuat menu utama
    menu_bar = tk.Menu(root)

    # Membuat sub-menu "File"
    file_menu = tk.Menu(menu_bar, tearoff=0)

    # Add commands to the Submenu
    file_menu.add_command(label="User Tutorial", command=user_tutorial)
    file_menu.add_command(label="About", command=about)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=on_closing)
    menu_bar.add_cascade(label="File", menu=file_menu)

    # Menambahkan menu bar ke root window
    root.config(menu=menu_bar)
    # Menjalankan aplikasi
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

    #Mengurutkan kolom "Date" secara descending saat aplikasi dimulai
    root.after(1000, lambda: sort_treeview_saham(tree_history_saham, "Date", True))
    root.after(1000, lambda: sort_treeview_keltner(tree_keltner_channel, "Date", True))

def check_internet_connection(url='https://finance.yahoo.com/', timeout=5):
    try:
        response = requests.get(url, timeout=timeout)
        # If the request succeeds, the internet connection is available
        return True
    except (requests.ConnectionError, requests.Timeout):
        # If the request fails, the internet connection is not available
        return False

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit from Algo Trading Tool?"):
        root.destroy()

def update_progress_bar(progress_bar, window):
    """Fungsi untuk memperbarui progress bar."""
    for i in range(101):
        # Perbarui nilai progress bar
        progress_bar['value'] = i
        
        # Perbarui GUI
        window.update_idletasks()
        
        # Tunggu sejenak
        time.sleep(0.01)

    # Setelah selesai, tutup window progress bar dan tampilkan form utama
    if check_internet_connection():
        ##print("Internet connection is available.")
        window.destroy()
        create_registration_widgets()
        
            
    else:
        ##print("No internet connection.")
        messagebox.showwarning("Error Message", "No internet connection.!!")
        window.destroy()
        

    #window.destroy()
    #create_main_window()

def create_progress_bar(width, height):
    """Fungsi untuk membuat window dengan progress bar."""
    progress_window = tk.Tk()

    ##--Center Screen--##
    screen_width = progress_window.winfo_screenwidth()
    screen_height = progress_window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    progress_window.geometry(f'{width}x{height}+{x}+{y}')

    progress_window.configure(bg='#a9a9a9')  # Set background color
    progress_window.overrideredirect(True)  # Remove window decorations

    progress_label = tk.Label(progress_window, bg='#a9a9a9', text="Progress...")
    progress_label.pack(pady=1)

    progress_bar = ttk.Progressbar(progress_window, orient='horizontal', length=450, mode='determinate')
    progress_bar.pack(pady=1)

    # Perbarui progress bar
    progress_window.after(100, update_progress_bar, progress_bar, progress_window)
    progress_window.mainloop()

# Memanggil fungsi untuk membuat tampilan utama
##create_main_window()
create_progress_bar(width=500, height=70)


