import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_csv("datasets/flo_data_20k.csv")
df = df_.copy()
df.head(10)
df.columns

"""
master_id Eşsiz müşteri numarası
order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
last_order_channel En son alışverişin yapıldığı kanal
first_order_date Müşterinin yaptığı ilk alışveriş tarihi
last_order_date Müşterinin yaptığı son alışveriş tarihi
last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
"""

#Betimsel İstatistik
#describe() fonksiyonu ile aykırılıkları gözlemleyebiliriz. Burada sayısal değişkenlere ait betimsel istatistikler yer almaktadır.
df.describe().T

#Boş değerler
df.isnull().sum() #boş değer yok.

#Değişken tipleri

for col in df.columns:
    print(df[col].dtype)

#Adım 1: Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturalım.

df["master_id"].nunique()

df["Total_Transaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Total_Price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.groupby("master_id").agg({"Total_Transaction": lambda x: x.sum(),
                             "Total_Price": lambda x: x.sum()})

#Adım 2: Değişken tiplerini inceleyelim. Tarih ifade eden değişkenlerin tipini date'e çevirelim.

for col in df.columns:
    print(df[col].dtype)

df.info()

for col in df.columns:
    if 'date' in col.lower() and df[col].dtype == 'object':
        df[col] = pd.to_datetime(df[col])
df.info()


#Adım 3: En fazla kazancı getiren ilk 10 müşteriyi sıralayalım.

en_fazla_kazanc = df.sort_values(by = "Total_Price", ascending = False).head(10)
print(en_fazla_kazanc)

#Adım 4: En fazla siparişi veren ilk 10 müşteriyi sıralayalım.

en_fazla_siparis = df.sort_values(by = "Total_Transaction", ascending = False).head(10)
print(en_fazla_siparis)

#Adım 5: Veri ön hazırlık sürecini fonksiyonlaştıralım.

def onhazirlik(dataframe, csv = False):

    df["Total_Transaction"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
    df["Total_Price"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
    df.groupby("master_id").agg({"Total_Transaction": lambda x: x.sum(),
                                 "Total_Price": lambda x: x.sum()})
    for col in df.columns:
        if 'date' in col.lower() and df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col])

    if csv:
        df.to_csv("flo_onhazirlik.csv")
    return df

df = df_.copy()
onhazirlik(df, csv = True)

############ RFM Metriklerinin Hesaplanması ############3

# Adım 1: Recency, Frequency ve Monetary.

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 1)
type(today_date)

rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'Total_Transaction': lambda Total_Transaction: Total_Transaction.sum(),
                                     'Total_Price': lambda Total_Price: Total_Price.sum()})

rfm.head()

rfm.columns = ['recency', 'frequency', 'monetary']

############## RFM Skorlarının Hesaplanması ###########

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.head()

############# RF Skorların Segment Olarak Tanımlanması ################

seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm.head()