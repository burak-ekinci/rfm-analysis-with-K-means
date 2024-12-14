import pandas as pd  # Veri manipülasyonu için
import matplotlib.pyplot as plt  # Veri görselleştirme
import seaborn as sns  # Veri görselleştirme
import datetime as dt  # Tarihsel işlemler
from sklearn.preprocessing import MinMaxScaler  # Min-max normalizasyonu
from yellowbrick.cluster import KElbowVisualizer  # K-elbow tekniği için yelowbrick aracı
from sklearn.cluster import KMeans  # K-means algoritması

# 1. Veri Yükleme ve Önışleme
# Veriyi yükle ve bir kopyasını oluştur
mainData = pd.read_csv("C:/Users/bltyc/Desktop/rfm/musteri.csv")
data = mainData.copy()
# Her ürünün toplam fiyatını hesapla
data["Total"] = data["Quantity"] * data["UnitPrice"]
print("emptys",data.isnull().sum())
# Eksik verileri ve mantıksız kayıtları temizle
data = data.dropna()
data = data.drop(data[data["Total"] <= 0].index)

sns.boxplot(data["Total"])
plt.show()

# 2. Aykırı Değerlerin Temizlenmesi (IQR Yöntemi)
Q1 = data["Total"].quantile(0.25)  # İlk çeyrek
Q3 = data["Total"].quantile(0.75)  # Üçüncü çeyrek
IQR = Q3 - Q1
lowerLimit = Q1 - 1.5 * IQR
upperLimit = Q3 + 1.5 * IQR
# Aykırı olmayan verileri filtrele
data = data[~((data["Total"] > upperLimit) | (data["Total"] < lowerLimit))]
data = data.reset_index(drop=True)  # İndeksi sıfırla

# 3. Veri Tipi Dönüşümü
# CustomerID'yi tam sayıya dönüştür
data["CustomerID"] = data["CustomerID"].astype("int")
# InvoiceDate sütununu datetime formatına dönüştür
data["InvoiceDate"] = pd.to_datetime(data["InvoiceDate"])
# Veri setindeki maksimum tarihi belirle
currentDate = dt.datetime(2011, 12, 9, 12, 50, 0)


# 4. RFM (Recency, Frequency, Monetary) Değerlerini Hesaplama
# Recency: Müşterinin son işlem tarihine göre gün sayısı
r = (currentDate - data.groupby("CustomerID")["InvoiceDate"].max()).apply(lambda x: x.days)
# Frequency: Fatura sayısı
tempFreq = data.groupby(["CustomerID", "InvoiceNo"]).size()
f = tempFreq.groupby("CustomerID").size()
# Monetary: Toplam harcama
m = data.groupby("CustomerID")["Total"].sum()

# RFM tablosunu oluştur
RFM = pd.DataFrame({"Recency": r, "Frequency": f, "Monetary": m}).reset_index()

# 5. Normalizasyon
# Veriyi 0-1 arasına sıkıştır
scaler = MinMaxScaler()
dfnorm = pd.DataFrame(scaler.fit_transform(RFM.iloc[:, 1:]), columns=RFM.columns[1:])

# 6. Optimal Küme Sayısını Belirleme (Elbow Tekniği)
elbowViz = KElbowVisualizer(KMeans(init="k-means++", random_state=0), k=(2, 10))
elbowViz.fit(dfnorm)
elbowViz.poof()

# 7. K-Means ile Kümeleme
optimalClusters = elbowViz.elbow_value_  # Elbow noktasındaki küme sayısı
groupModel = KMeans(n_clusters=optimalClusters, init="k-means++", random_state=0)
groupModel.fit(dfnorm)
RFM["Labels"] = groupModel.labels_

# 8. Grupları Analiz Etme ve Görselleştirme
# Grupların ortalama değerlerini gör
print(RFM.groupby("Labels").mean())
# Grupların dağılımını görselleştir
sns.scatterplot(x="Recency", y="Frequency", hue="Labels", data=RFM, palette="deep")
plt.show()
