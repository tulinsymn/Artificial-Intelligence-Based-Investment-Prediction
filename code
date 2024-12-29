import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import warnings
import time

# Suppress warnings and reduce TensorFlow logging verbosity
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Load datasets for USD, EUR, and Gold
dolar_data = pd.read_csv("duzenlenmis_veri_dolar1.csv")
euro_data = pd.read_csv("duzenlenmis_veri_euro1.csv")
altin_data = pd.read_csv("duzenlenmis_veri_altın1.csv")

# Preprocess each dataset
for data in [dolar_data, euro_data, altin_data]:
    # Convert date column to datetime format
    data["Tarih"] = pd.to_datetime(data["Tarih"], format="%d.%m.%Y", errors="coerce")
    # Extract year from the date column
    data["Yıl"] = data["Tarih"].dt.year

# Group data by year and calculate the yearly average
dolar_yearly = dolar_data.groupby("Yıl")["Şimdi"].mean().reset_index()
euro_yearly = euro_data.groupby("Yıl")["Şimdi"].mean().reset_index()
altin_yearly = altin_data.groupby("Yıl")["Şimdi"].mean().reset_index()

# Prompt user to select an investment type
yatirim_araci = ""
while yatirim_araci not in ["dolar", "euro", "altın"]:
    yatirim_araci = input("Yatırım aracını seçiniz (dolar/euro/altın): ").lower()
    if yatirim_araci not in ["dolar", "euro", "altın"]:
        print("Hatalı yatırım aracı seçildi! Lütfen doğru bir yatırım aracı giriniz.")

# Select the appropriate dataset based on the user's choice
if yatirim_araci == "dolar":
    data = dolar_yearly
elif yatirim_araci == "euro":
    data = euro_yearly
elif yatirim_araci == "altın":
    data = altin_yearly

# Display available years in the dataset
print(f"Mevcut veri yılları: {data['Yıl'].min()} - {data['Yıl'].max()}")

# Prompt user to enter a valid starting year
baslangic_yili = ""
while not baslangic_yili.isdigit() or int(baslangic_yili) < data['Yıl'].min() or int(baslangic_yili) > data['Yıl'].max():
    baslangic_yili = input("Başlangıç yılını giriniz: ")
    if not baslangic_yili.isdigit() or int(baslangic_yili) < data['Yıl'].min() or int(baslangic_yili) > data['Yıl'].max():
        print("Geçersiz yıl girişi! Lütfen mevcut yıllar arasında bir yıl giriniz.")

baslangic_yili = int(baslangic_yili)

# Prompt user to enter the initial investment amount
sermaye = ""
while True:
    sermaye = input("Yatırım için kullanılacak sermaye miktarını giriniz: ")
    try:
        sermaye = float(sermaye)
        break
    except ValueError:
        print("Geçersiz sermaye değeri! Lütfen geçerli bir sayı giriniz.")

# Filter data for training
filtered_data = data[data["Yıl"] >= (baslangic_yili - 4)]

# Normalize data to a range of [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(filtered_data["Şimdi"].values.reshape(-1, 1))

# Prepare the dataset for the LSTM model
X, y = [], []
for i in range(4, len(scaled_data)):
    X.append(scaled_data[i-4:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Use the entire dataset for training
X_train = X
y_train = y

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], 1)))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mean_squared_error")

print("Veriler işleniyor, lütfen bekleyiniz...")

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=0)

print("İşlem tamamlandı!")

# Predict future values (up to 2030)
future_years = np.arange(filtered_data["Yıl"].max() + 1, 2031)
predictions = []

last_4_years = scaled_data[-4:].reshape(1, -1)
for year in future_years:
    prediction = model.predict(last_4_years, verbose=0)
    predictions.append(prediction[0, 0])
    last_4_years = np.append(last_4_years[:, 1:], prediction).reshape(1, -1)

# Rescale predictions back to original scale
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

# Combine historical and predicted data
future_data = pd.DataFrame({"Yıl": future_years, "Şimdi": predictions.flatten()})
combined_data = pd.concat([filtered_data, future_data])

# Calculate investment value based on predictions
combined_data["Yatırım Değeri"] = sermaye * (combined_data["Şimdi"] / combined_data["Şimdi"].iloc[0])
combined_data = combined_data.dropna(subset=["Yatırım Değeri"])
combined_data = combined_data.reset_index(drop=True)

combined_data["Yatırım Değeri"] = 0
for index, row in combined_data.iterrows():
    if row["Yıl"] >= baslangic_yili:
        combined_data.loc[index, "Yatırım Değeri"] = sermaye * (row["Şimdi"] / combined_data["Şimdi"].loc[combined_data["Yıl"] == baslangic_yili].values[0])

# Display results
print("\nGeçmiş ve Gelecek Tahminleri:")
pd.options.display.float_format = '{:,.2f}'.format
print(combined_data[["Yıl", "Şimdi", "Yatırım Değeri"]])

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(combined_data["Yıl"], combined_data["Yatırım Değeri"], marker="o", label="Yatırım Getirisi")
plt.title(f"{yatirim_araci.capitalize()} Yatırım Getirisi ve Tahmin", fontsize=14)
plt.xlabel("Yıl", fontsize=12)
plt.ylabel("Yatırım Değeri (TL)", fontsize=12)
plt.grid()
plt.legend()
years_range = np.arange(combined_data["Yıl"].min(), combined_data["Yıl"].max() + 1, 3)
plt.xticks(years_range)
plt.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=True, labelleft=True)
plt.show()

final_investment_value_2030 = sermaye * (predictions[-1][0] / combined_data["Şimdi"].loc[combined_data["Yıl"] == baslangic_yili].values[0])
print(f"\n{baslangic_yili} yılında yatırdığınız {sermaye} TL'nin 2030 yılına kadar tahmin edilen değeri: {final_investment_value_2030:,.2f} TL")
