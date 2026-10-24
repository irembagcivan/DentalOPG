# DentalOPG

Panoramik diş röntgenlerinde (OPG) derin öğrenme ile diş ve diş çevresi bulgularının otomatik tespiti projesi.

## Proje Hakkında

Panoramik radyografiler, diş hekimliğinde tek görüntüde tüm çene yapısını değerlendirmeye olanak sağlayan yaygın bir görüntüleme yöntemidir. Bu projenin amacı, **YOLOv8** (nesne tespiti) ve **U-Net** (segmentasyon) modellerini kullanarak bu görüntülerde çürük, dolgu, kanal tedavisi, gömülü diş, eksik diş gibi bulguları otomatik olarak tespit eden bir sistem geliştirmektir.

Proje aşamalıdır:

1. Veri seti analizi, kalite kontrolü, veri temizliği ve ön işleme *(mevcut aşama)*
2. Model eğitimi
3. Model değerlendirmesi ve sonuçların yorumlanması

## Amaç

- Panoramik röntgenlerde 14 farklı sınıfı tespit edebilen bir model geliştirmek
- Veri setinin kalitesini ve etiket tutarlılığını eğitimden önce sistematik olarak incelemek
- Sınıf dengesizliği gibi veri kaynaklı sorunları belirleyip çözüm stratejileri geliştirmek

## Veri Seti

- DentalOPG-1550: 1525 panoramik diş röntgeni, YOLOv8 formatında etiketlenmiş
- Üç kümeye ayrılmıştır: `train`, `valid`, `test`
- 14 sınıf, sınırlayıcı kutu (bounding box) etiketleri
- Kaynak ve lisans: [Mendeley Data](https://data.mendeley.com/datasets/rtt726b26d/3)

> Veri seti ve ondan türetilen klasörler (`14 class DENTAL.v4i.yolov8`, `14 class DENTAL.v4i.yolov8_preprocessed`, `karantina_bos_dosyalar`) bu depoda paylaşılmamıştır.

### Sınıflar

BDR, Bridge, Cap, Caries, Distoangular impaction, Filling, Horizontal Impaction, Missing Teeth, Mixed Dentition, Normal Teeth, Periodontitis, Rootcanal, Unerupted Wisdom Teeth, Apical abscess

## Bu Depodaki Çalışmalar

| Dosya | Açıklama |
|---|---|
| `01_veri_seti_analizi.py` | Görüntü-etiket eşleşmesi, etiket formatı (kutu/poligon) ve sınıf dağılımının analizi |
| `02_kalite_etiket_kontrolu.py` | Poligon etiketlerin kutuya dönüştürülmesi, bulanıklık / parlaklık / kontrast taraması ve etiketlerin görsel doğrulaması |
| `03_bos_dosyalari_ayikla.py` | Boş etiketli dosyaların ve eşleşen görüntülerin karantina klasörüne taşınması |
| `04_gurultu_azaltma_clahe.py` | Görüntülere gürültü azaltma, CLAHE ve hafif keskinleştirme uygulanması |
| `05_lezyon_crop_kontrolu.py` | Lezyon bölgelerinin kırpılıp orijinal ve ön işlenmiş halinin karşılaştırılması |
| `goruntu_kalite_raporu.csv` | 1525 görüntü için kalite metrikleri (Laplacian varyansı, parlaklık, kontrast) |
| `goruntu_kalite_raporu_YENILENMIS.xlsx` | Görüntü kalite raporunun güncellenmiş Excel sürümü |
| `goruntu_kaliteleri.png` | Görüntü kalite metriklerinin dağılım grafikleri |
| `ground_truth_kontrol_ornekleri.png` | Etiketlerin doğrulandığı örnek görüntüler |
| `lezyon_crop_karsilastirma.png` | Dolgu, kaplama ve çürük kırpmalarında orijinal / eski / yeni ön işleme karşılaştırması |
| `lezyon_crop_karsilastirma_duzeltilmis.png` | İnce bulgularda (çürük, periodontitis, apikal apse, kanal tedavisi) aynı karşılaştırma |

## Yöntem

- **Veri analizi:** Pandas ile küme ve sınıf bazlı istatistikler
- **Görüntü kalitesi:** OpenCV ile Laplacian varyansı (bulanıklık), ortalama parlaklık ve standart sapma (kontrast)
- **Etiket doğrulama:** Ground truth kutularının görüntü üzerine çizdirilerek kontrolü
- **Ön işleme:** Bilateral filtre (gürültü azaltma), CLAHE (lokal kontrast iyileştirme) ve hafif keskinleştirme
- **Ön işleme doğrulaması:** Lezyon bölgelerinin kırpılarak orijinal ve işlenmiş görüntülerle görsel karşılaştırılması
- **Modeller (planlanan):** YOLOv8 (tespit), U-Net (segmentasyon)

## Kullanılan Teknolojiler

Python, OpenCV, NumPy, Pandas, Matplotlib, PyYAML, Ultralytics YOLOv8

## Kurulum ve Çalıştırma

```bash
pip install opencv-python numpy pandas matplotlib pyyaml
python 01_veri_seti_analizi.py
python 02_kalite_etiket_kontrolu.py
python 03_bos_dosyalari_ayikla.py
python 04_gurultu_azaltma_clahe.py
python 05_lezyon_crop_kontrolu.py
```

Betiklerdeki `VERI_SETI_DIZINI` değişkenini kendi bilgisayarınızdaki veri seti yoluna göre güncelleyin.

## Proje Durumu

Model eğitimi öncesi Keşifçi Veri Analizi (EDA), veri temizliği ve ön işleme aşamaları büyük ölçüde tamamlanmıştır. Veri setinin yapısal analizi yapılmış, format uyumsuzlukları giderilmiş, boş etiketli dosyalar ayrılmış ve görüntülere ön işleme uygulanmıştır. Sıradaki ana adım, ön işlemenin model başarısına etkisinin ölçülmesidir.

### 1. `01_veri_seti_analizi.py` — Veri Seti Keşfi ve Format Kontrolü

Veri setinin genel sağlığını ve etiketleme karakteristiklerini haritalandırmak için kullanıldı.

**Yapılan işlemler**

- `train`, `valid` ve `test` kümelerindeki `images` ve `labels` klasörleri taranarak eşleşmeyen (yetim) görüntü veya etiket dosyası olup olmadığı denetlendi. Eşleşme oranı %100 olarak doğrulandı.
- 14 sınıfa ait etiket dosyaları (`.txt`) satır satır okunarak format analizi yapıldı.
- Her satırın sınırlayıcı kutu (4 koordinat) mu yoksa poligon maske (6 ve üzeri koordinat) mi olduğu hesaplandı.

**Temel bulgular**

- 14.367 etiketin neredeyse tamamı kutu (bbox) formatındadır. Yalnızca `Normal Teeth` sınıfına ait 130 poligon vardır.
- Çürük, apse veya gömülü diş gibi patolojik hedeflerin hiçbirinde poligon maskesi yoktur. Bu nedenle U-Net segmentasyonu için maskelerin SAM (Segment Anything Model) ile otomatik üretilmesine karar verilmiştir.

### 2. `02_kalite_etiket_kontrolu.py` — Görüntü Kalite Analizi ve Format Düzeltme

Analiz aşamasında tespit edilen sorunları gidermek ve eğitim performansını düşürebilecek (gürültülü/bulanık) görüntüleri istatistiksel olarak belirlemek için kullanıldı.

**Yapılan işlemler**

- **Format standardizasyonu:** 5 karışık dosyadaki 130 poligon etiketi, kendilerini çevreleyen en küçük sınırlayıcı kutuya dönüştürüldü. Veri seti %100 standart YOLO kutu formatına getirildi.
- **Kalite ve bulanıklık analizi:** 1525 görüntünün tamamı için Laplacian varyansı (bulanıklık), ortalama parlaklık ve standart sapma (kontrast) hesaplandı ve `goruntu_kalite_raporu.csv` olarak kaydedildi.
- **Boş etiket (label noise) tespiti:** Hiç kutu içermeyen, etiketlenmesi atlanmış 22 boş `.txt` dosyası tespit edilip raporlandı.
- **Ground truth görselleştirme:** Rastgele seçilen röntgenlerin üzerine etiketler renkli kutular olarak çizdirildi (`ground_truth_kontrol_ornekleri.png`).

**Temel bulgular**

- Laplacian skoru 25'in altında kalan aşırı bulanık görüntüler ve skoru 500'ün üzerinde olan aşırı kumlu (gürültülü) görüntüler belirlendi.
- Bazı röntgenlerde `Normal Teeth` (sağlıklı dişler) sınıfının bilerek etiketlenmediği görüldü.
- SAM'e geçmeden önce röntgenlere gürültü azaltma ve kontrast iyileştirme uygulanmasının gerekli olduğu doğrulandı.

### 3. `03_bos_dosyalari_ayikla.py` — Boş Etiketli Dosyaların Ayıklanması

02 numaralı betikte tespit edilen boş etiketli dosyaları eğitim verisinden ayırmak için kullanıldı.

**Yapılan işlemler**

- `train`, `valid` ve `test` kümelerindeki `labels` klasörleri taranarak hiç satır içermeyen `.txt` dosyaları belirlendi.
- Bu etiket dosyaları, eşleşen görüntüleriyle birlikte `karantina_bos_dosyalar/` klasörüne (`images` ve `labels` alt klasörleri) taşındı. Dosyalar silinmedi, gerektiğinde geri alınabilir.
- 02'de tespit edilen 22 boş etiketli dosya veri setinden izole edildi.

### 4. `04_gurultu_azaltma_clahe.py` — Görüntü Ön İşleme

Düşük kontrast ve gürültü sorunlarını gidermek için görüntülere sırasıyla şu adımlar uygulandı:

1. **Bilateral filtre:** Kenarları koruyarak gürültüyü azaltır.
2. **CLAHE:** Lokal kontrastı artırır, kemik ve diş yapılarını belirginleştirir.
3. **Hafif keskinleştirme (unsharp mask):** Gürültü azaltma ve kontrast işlemlerinde yumuşayan ince detayları toparlar.

İşlenmiş görüntüler, orijinal veriyi bozmamak için ayrı bir klasöre (`14 class DENTAL.v4i.yolov8_preprocessed`) yazıldı.

**Parametre ayarı:** İlk denemede bilateral filtre fazla güçlü uygulanmıştı (`d=9`, `sigma=75`). Diş aralarındaki ince çizgiler ve kök detayları yumuşadığı için filtre hafifletildi (`d=5`) ve keskinleştirme adımı eklendi.

### 5. `05_lezyon_crop_kontrolu.py` — Ön İşlemenin Lezyon Bölgelerinde Doğrulanması

Ön işlemenin ince bulgularda detay kaybettirip kaybettirmediğini görmek için etiketli bölgeler kırpılarak orijinal ve işlenmiş halleri yan yana çizdirildi. Kırpmalar `train` kümesinden rastgele seçildi ve kutunun çevresinden pay bırakıldı.

İki farklı parametre seti karşılaştırıldı:

| | Bilateral sigma | CLAHE clipLimit | Keskinleştirme |
|---|---|---|---|
| **Eski** | 25 | 2.0 | 0.3 |
| **Yeni (dengeli)** | 30 | 1.5 | 0.15 |

**Temel bulgular**

- Ön işleme tüm sınıflarda diş, kök ve kemik sınırlarının kontrastını artırdı. Orijinalde neredeyse düz görünen periodontitis ve apikal apse bölgeleri işlenmiş görüntülerde belirgin hale geldi.
- Parlak yapılarda (dolgu, kaplama) güçlü kontrast, iç dokuyu düzleştirip parlak alanı doyurabiliyor. Bu nedenle çok yüksek kontrast ayarları riskli.
- Eski ayar daha yüksek kontrast veriyor (soluk bulgularda yararlı, parlak yapılarda daha agresif). Yeni ayar daha doğal görünüyor. İkisi arasındaki fark küçük ve gözle kesin karar vermek zor.
- Bazı kırpmaların çözünürlüğü yalnızca birkaç düzine piksel. Çok küçük lezyonlarda ön işleme bu sınırı aşamıyor.
- Hangi ayarın daha iyi olduğuna gözle değil, model başarısıyla karar verilecek.

## Sıradaki Adımlar

- [x] **Boş etiketli dosyaların izole edilmesi:** 22 dosya karantinaya alındı
- [x] **Ön işleme:** CLAHE, gürültü azaltma ve hafif keskinleştirme uygulandı
- [x] **Ön işlemenin görsel doğrulaması:** Lezyon kırpmalarıyla orijinal ve işlenmiş görüntüler karşılaştırıldı
- [x] **Aşırı bulanık / gürültülü görüntülerin değerlendirilmesi:** Laplacian eşiklerine göre eğitim setinden çıkarılıp çıkarılmayacağına karar verilmesi
- [x] **Ön işlemenin model üzerindeki etkisinin ölçülmesi:** Orijinal ve işlenmiş veriyle aynı ayarlarda kısa YOLOv8 eğitimleri yapılıp sınıf bazlı (özellikle Caries) mAP karşılaştırılması
- [ ] **SAM ile maske üretimi:** Bbox koordinatlarının SAM'e ipucu olarak verilip U-Net eğitimi için çok sınıflı (multi-class) ve ikili (binary) maskelerin otomatik üretilmesi
- [ ] **Model eğitimi:** YOLOv8 ve U-Net eğitimi
- [ ] **Değerlendirme:** Sonuçların yorumlanması
