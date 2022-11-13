##############################################################
# FLO: RFM Analizi ile Müşteri Segmentasyonu
##############################################################

# Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri
# belirlemek istiyor. Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere
# göre gruplar oluşturulacak.
# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online
# hem offline alışveriş yapan) olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden
# oluşmaktadır.

#####################################
# Features
#####################################
# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihiæ
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import squarify
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##############################################################
# Task 1: Veriyi Anlama ve Hazırlama
##############################################################

# Step 1: flo_data_20K.csv verisini okuyunuz. Dataframe’in kopyasını oluşturunuz

df_ = pd.read_csv("flo_data_20k.csv")

df = df_.copy()

# Step 2: Veri setinde
#         a. İlk 10 gözlem,
#         b. Değişken isimleri,
#         c. Betimsel istatistik,
#         d. Boş değer,
#         e. Değişken tipleri, incelemesi yapınız

df.head()
df.describe().T
df.isnull().sum()
df.dtypes
df.nunique()

df["order_channel"].value_counts()
df["last_order_channel"].value_counts()

# Step 3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
#         Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz

# Omnichannel değeri var mı?
df[df["order_channel"] == "Omnichannel"]["master_id"].count()

# For omnichannel total over {online + offline}
df["total_of_purchases"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

# Total cost for omnichannel (toplam alınan ürün sayısı)
df["total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

# Step 4:  Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
df.dtypes
convert =["first_order_date","last_order_date","last_order_date_online","last_order_date_offline"]
df[convert] = df[convert].apply(pd.to_datetime)

df.head()

# Step 5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız

# order channel , total of purchase and total expenditure distribution
df.groupby('order_channel').agg({'total_of_purchases':'sum',
                                    'total_expenditure':'count'}).sort_values(by='total_expenditure', ascending=False)


# Step 6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız
df.groupby("master_id").agg({"total_expenditure":"sum"}).sort_values(by="total_expenditure", ascending=False).head(10)


# Step 7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız
df.groupby('master_id').agg({'total_of_purchases': 'sum'}).\
    sort_values(by='total_of_purchases', ascending=False).head(10)

# Step 8: Veri ön hazırlık sürecini fonksiyonlaştırınız.
def data_preparation_process(df):
    # For omnichannel total over {online + offline}
    df['total_of_purchases'] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]

    # Total cost for omnichannel
    df["total_expenditure"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

    # Converting the above mentioned column types from object to datetime format
    convert = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
    df[convert] = df[convert].apply(pd.to_datetime)

    return df

##############################################################
# Task 2: RFM Metriklerinin Hesaplanması
##############################################################

# Step 1: Recency, Frequency ve Monetary tanımlarını yapınız.
last_date = df["last_order_date"].max()   #2021-05-30

# Analizin yapıldığı tarih
today_date = dt.datetime(2021, 6, 2)

# Step 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız
rfm = df.groupby('master_id').agg({'last_order_date': lambda last_order_date: (today_date - last_order_date.max()).days,
                                     'total_of_purchases': lambda total_of_purchases: total_of_purchases.sum(),
                                     'total_expenditure': lambda total_expenditure: total_expenditure.sum()})

rfm.columns = ['recency', 'frequency', 'monetary']

rfm.describe().T

##############################################################
# Task 3: RF Skorunun Hesaplanması
##############################################################

# Step 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
# Step 2: Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
rfm.head()

# Step 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.
rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm[rfm["RF_SCORE"] == "55"].head()


##############################################################
# Task 4: RF Skorunun Segment Olarak Tanımlanması
##############################################################

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

# RFM_SCORE değişkenindeki değerleri seg_map değerleriyle değiştir
# Bunu da yeni bir dğeişken(segment) oluşturup ona ata
rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)
rfm.head()


##############################################################
# Task 5: Aksiyon Zamanı !
##############################################################

# Step 1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz
rfm.groupby("segment").agg({"recency":["mean","count"], "frequency":["mean","count"], "monetary":["mean","count"]})


# Step 2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
# yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz

# master_id index'ten değişkene çevir
loyal = rfm[(rfm["segment"]=="champions") | (rfm["segment"]=="loyal_customers")].reset_index()
woman = df[df["interested_in_categories_12"].str.contains("KADIN")]
loyal_woman = pd.merge(loyal, woman[["interested_in_categories_12", "master_id"]], on="master_id")
loyal_woman.to_csv(path_or_buf="loyal_woman.csv")


# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz

new_segment = rfm[(rfm["segment"]=="cant_loose") | (rfm["segment"]=="hibernating") |
    (rfm["segment"]=="about_to_sleep") | (rfm["segment"]=="new_customers")].reset_index()

male_kids = df[df["interested_in_categories_12"].str.contains("ERKEK|COCUK")]

new_segment_male = pd.merge(new_segment, male_kids[["interested_in_categories_12", "master_id"]], on="master_id")
new_segment_male.to_csv(path_or_buf="new_segment_male.csv")



