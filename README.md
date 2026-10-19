# DentalOPG

Panoramik diş röntgenlerinde (OPG) derin öğrenme ile diş ve diş çevresi bulgularının otomatik tespiti projesi.

## Proje Hakkında

Panoramik radyografiler, diş hekimliğinde tek görüntüde tüm çene yapısını değerlendirmeye olanak sağlayan yaygın bir görüntüleme yöntemidir. Bu projenin amacı, YOLOv8 nesne tespit modelini kullanarak bu görüntülerde çürük, dolgu, kanal tedavisi, gömülü diş, eksik diş gibi bulguları otomatik olarak tespit eden bir sistem geliştirmektir.

Proje aşamalıdır:

1. Veri seti analizi ve ön işleme
2. Model eğitimi
3. Model değerlendirmesi ve sonuçların yorumlanması

## Amaç

- Panoramik röntgenlerde 14 farklı sınıfı tespit edebilen bir model geliştirmek
- Veri setinin kalitesini ve etiket tutarlılığını eğitimden önce sistematik olarak incelemek
- Sınıf dengesizliği gibi veri kaynaklı sorunları belirleyip çözüm stratejileri geliştirmek

## Veri Seti

- Panoramik diş röntgenlerinden oluşan, YOLOv8 formatında etiketlenmiş bir veri seti
- Üç kümeye ayrılmıştır: train, valid, test
- 14 sınıf, sınırlayıcı kutu (bounding box) etiketleri
- Kaynak ve lisans: *(https://data.mendeley.com/datasets/rtt726b26d/3)*

> Veri seti bu depoda paylaşılmamıştır.

## Tespit Edilen Sınıflar

BDR, Bridge, Cap, Caries, Distoangular impaction, Filling, Horizontal Impaction, Missing Teeth, Mixed Dentition, Normal Teeth, Periodontitis, Rootcanal, Unerupted Wisdom Teeth, Apical abscess

## Bu Depodaki Çalışmalar

| Dosya | Açıklama |
|---|---|
| `01_veri_seti_analizi.py` | Görüntü-etiket eşleşmesi, etiket formatı (kutu/poligon) ve sınıf dağılımının analizi |
| `02_kalite_etiket_kontrolu.py` | Poligon etiketlerin kutuya dönüştürülmesi, bulanıklık / parlaklık / kontrast taraması ve etiketlerin görsel doğrulaması |
| `goruntu_kaliteleri.png` | Görüntü kalite metriklerinin dağılım grafikleri |
| `ground_truth_kontrol_ornekleri.png` | Etiketlerin doğrulandığı örnek görüntüler |

## Yöntem

- **Veri analizi:** Pandas ile küme ve sınıf bazlı istatistikler
- **Görüntü kalitesi:** OpenCV ile Laplacian varyansı (bulanıklık), ortalama parlaklık ve standart sapma (kontrast)
- **Etiket doğrulama:** Ground truth kutularının görüntü üzerinde çizdirilerek kontrolü
- **Model (planlanan):** YOLOv8

## Kullanılan Teknolojiler

Python, OpenCV, NumPy, Pandas, Matplotlib, PyYAML, Ultralytics YOLOv8

## Kurulum ve Çalıştırma

```bash
pip install opencv-python numpy pandas matplotlib pyyaml
python 01_veri_seti_analizi.py
python 02_kalite_etiket_kontrolu.py
```

Betiklerdeki `VERI_SETI_DIZINI` değişkenini kendi bilgisayarınızdaki veri seti yoluna göre güncelleyin.

## Proje Durumu

- [x] Veri seti yapısının ve etiket formatlarının analizi
- [x] Görüntü kalitesi taraması
- [x] Etiketlerin görsel doğrulaması
- [ ] Veri setinin yeniden bölünmesi
- [ ] Model eğitimi
- [ ] Değerlendirme ve raporlama

