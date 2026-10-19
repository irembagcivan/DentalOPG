import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

VERI_SETI_DIZINI = Path("/Users/irembagcivan/Desktop/DentalOPG/14 class DENTAL.v4i.yolov8")
ALT_KUMELER = ["train", "valid", "test"]
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp"}

def sinif_isimlerini_yukle(yaml_yolu):
    if yaml_yolu.exists():
        with open(yaml_yolu, "r", encoding="utf-8") as dosya:
            return yaml.safe_load(dosya).get("names", [])
    return []

# 1. KARISIK DOSYALARDAKİ POLİGONLARI KUTUYA (BBOX) DÖNÜŞTÜRME
def poligonlari_kutuya_donustur(veri_seti_dizini):
    """
    train kümesindeki 5 karışık dosyada bulunan poligon koordinatlarını 
    YOLOv8 sınırlayıcı kutu (merkez_x, merkez_y, genislik, yukseklik) formatına çevirir.
    """
    etiket_dizini = veri_seti_dizini / "train" / "labels"
    donusturulen_dosya_sayisi = 0
    donusturulen_poligon_sayisi = 0

    for etiket_yolu in etiket_dizini.glob("*.txt"):
        with open(etiket_yolu, "r", encoding="utf-8") as dosya:
            satirlar = [s.strip() for s in dosya.readlines() if s.strip()]
        yeni_satirlar = []
        degisiklik_yapildi = False

        for satir in satirlar:
            parcalar = satir.split()
            sinif_id = parcalar[0]
            koordinatlar = [float(x) for x in parcalar[1:]]

            # Eğer satır poligon ise (6 veya daha fazla koordinat)
            if len(koordinatlar) >= 6 and len(koordinatlar) % 2 == 0:
                x_ler = koordinatlar[0::2]
                y_ler = koordinatlar[1::2]
                min_x, max_x = min(x_ler), max(x_ler)
                min_y, max_y = min(y_ler), max(y_ler)

                merkez_x = (min_x + max_x) / 2.0
                merkez_y = (min_y + max_y) / 2.0
                genislik = max_x - min_x
                yukseklik = max_y - min_y

                yeni_satir = f"{sinif_id} {merkez_x:.6f} {merkez_y:.6f} {genislik:.6f} {yukseklik:.6f}"
                yeni_satirlar.append(yeni_satir)
                degisiklik_yapildi = True
                donusturulen_poligon_sayisi += 1
            else:
                yeni_satirlar.append(satir)

        if degisiklik_yapildi:
            with open(etiket_yolu, "w", encoding="utf-8") as dosya:
                dosya.write("\n".join(yeni_satirlar) + "\n")
            donusturulen_dosya_sayisi += 1

    print(f"[BİLGİ] {donusturulen_dosya_sayisi} dosyadaki toplam {donusturulen_poligon_sayisi} poligon standart kutuya dönüştürüldü.")

# 2. OPENCV İLE TÜM GÖRÜNTÜLERİN KALİTE VE BULANIKLIK TARAMASI
def goruntu_kalitesini_tara(veri_seti_dizini, alt_kumeler):
    """
    Tüm röntgenlerin Laplacian bulanıklık skorunu, parlaklık ve kontrast (standart sapma) 
    değerlerini hesaplar; boş etiketli dosyaları işaretler.
    """
    kalite_kayitlari = []

    for kume in alt_kumeler:
        goruntu_dizini = veri_seti_dizini / kume / "images"
        etiket_dizini = veri_seti_dizini / kume / "labels"

        for goruntu_yolu in goruntu_dizini.iterdir():
            if goruntu_yolu.suffix.lower() not in GORUNTU_UZANTILARI:
                continue

            etiket_yolu = etiket_dizini / f"{goruntu_yolu.stem}.txt"
            etiket_sayisi = 0
            if etiket_yolu.exists():
                with open(etiket_yolu, "r", encoding="utf-8") as f:
                    etiket_sayisi = len([s for s in f.readlines() if s.strip()])

            goruntu = cv2.imread(str(goruntu_yolu), cv2.IMREAD_GRAYSCALE)
            if goruntu is None:
                kalite_kayitlari.append({
                    "Alt Küme": kume, "Dosya Adı": goruntu_yolu.name,
                    "Bulanıklık Skoru": 0, "Ortalama Parlaklık": 0,
                    "Kontrast (Std)": 0, "Etiket Sayısı": etiket_sayisi, "Durum": "BOZUK_GORUNTU"
                })
                continue

            bulaniklik_skoru = cv2.Laplacian(goruntu, cv2.CV_64F).var()
            ortalama_parlaklik = np.mean(goruntu)
            kontrast_std = np.std(goruntu)

            durum = "NORMAL"
            if etiket_sayisi == 0:
                durum = "BOS_ETIKET"

            kalite_kayitlari.append({
                "Alt Küme": kume,
                "Dosya Adı": goruntu_yolu.name,
                "Bulanıklık Skoru": round(bulaniklik_skoru, 2),
                "Ortalama Parlaklık": round(ortalama_parlaklik, 2),
                "Kontrast (Std)": round(kontrast_std, 2),
                "Etiket Sayısı": etiket_sayisi,
                "Durum": durum
            })

    tablo = pd.DataFrame(kalite_kayitlari)
    
    # Bulanıklık eşiği için alt %5'lik dilimi (en bulanık görüntüler) belirleme
    alt_yuzde_5_esigi = tablo["Bulanıklık Skoru"].quantile(0.05)
    print("\n=== GÖRÜNTÜ KALİTESİ İSTATİSTİKLERİ ===")
    print(tablo[["Bulanıklık Skoru", "Ortalama Parlaklık", "Kontrast (Std)"]].describe().round(2))
    print(f"\n[ANALİZ] En bulanık %5'lik dilim sınır değeri (Laplacian): {alt_yuzde_5_esigi:.2f}")

    print("\n=== EN BULANIK 10 RÖNTGEN GÖRÜNTÜSÜ ===")
    en_bulaniklar = tablo.sort_values(by="Bulanıklık Skoru", ascending=True).head(10)
    print(en_bulaniklar.to_string(index=False))

    bos_etiketliler = tablo[tablo["Durum"] == "BOS_ETIKET"]
    print(f"\n=== BOŞ ETİKETLİ DOSYALAR (Toplam: {len(bos_etiketliler)}) ===")
    print(bos_etiketliler[["Alt Küme", "Dosya Adı", "Bulanıklık Skoru"]].to_string(index=False))

    # Raporu CSV olarak kaydetme (Danışman hocaya sunmak için)
    tablo.to_csv("goruntu_kalite_raporu.csv", index=False, encoding="utf-8-sig")
    print("\n[BİLGİ] Tüm görüntülerin kalite raporu 'goruntu_kalite_raporu.csv' olarak kaydedildi.")
    return tablo

# 3. GROUND TRUTH GÖRSEL DOĞRULAMA (ÖRNEK 4 GÖRÜNTÜ ÇİZDİRME)
def ornek_etiketleri_cizdir(veri_seti_dizini, sinif_isimleri, ornek_sayisi=4):
    """train kümesinden örnek röntgenler seçerek Ground Truth kutularını görselleştirir."""
    goruntu_dizini = veri_seti_dizini / "train" / "images"
    etiket_dizini = veri_seti_dizini / "train" / "labels"
    goruntu_listesi = [p for p in goruntu_dizini.iterdir() if p.suffix.lower() in GORUNTU_UZANTILARI][:ornek_sayisi]

    np.random.seed(42)
    renkler = np.random.randint(60, 255, size=(len(sinif_isimleri), 3)).tolist()

    sekil, eksenler = plt.subplots(2, 2, figsize=(15, 10))
    eksenler = eksenler.flatten()

    for i, goruntu_yolu in enumerate(goruntu_listesi):
        goruntu = cv2.imread(str(goruntu_yolu))
        goruntu = cv2.cvtColor(goruntu, cv2.COLOR_BGR2RGB)
        yukseklik, genislik = goruntu.shape[:2]
        etiket_yolu = etiket_dizini / f"{goruntu_yolu.stem}.txt"

        # Kutular çizilmeden ÖNCE, CSV ile birebir aynı yöntemle hesapla
        ham_gri_goruntu = cv2.imread(str(goruntu_yolu), cv2.IMREAD_GRAYSCALE)
        bulaniklik = cv2.Laplacian(ham_gri_goruntu, cv2.CV_64F).var()

        if etiket_yolu.exists():
            with open(etiket_yolu, "r", encoding="utf-8") as f:
                satirlar = [s.strip() for s in f.readlines() if s.strip()]

            for satir in satirlar:
                parcalar = satir.split()
                sinif_id = int(parcalar[0])
                koordinatlar = [float(x) for x in parcalar[1:]]
                renk = renkler[sinif_id % len(renkler)]
                sinif_adi = sinif_isimleri[sinif_id] if sinif_id < len(sinif_isimleri) else str(sinif_id)

                if len(koordinatlar) == 4:
                    mx, my, kg, ky = koordinatlar
                    x1, y1 = int((mx - kg / 2) * genislik), int((my - ky / 2) * yukseklik)
                    x2, y2 = int((mx + kg / 2) * genislik), int((my + ky / 2) * yukseklik)
                    cv2.rectangle(goruntu, (x1, y1), (x2, y2), renk, 2)
                    cv2.putText(goruntu, sinif_adi, (x1, max(12, y1 - 4)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, renk, 1, cv2.LINE_AA)

        eksenler[i].imshow(goruntu)
        eksenler[i].set_title(f"{goruntu_yolu.name[:25]}... | Blur: {bulaniklik:.1f}")
        eksenler[i].axis("off")

    plt.tight_layout()
    plt.savefig("ground_truth_kontrol_ornekleri.png", dpi=200)
    print("[BİLGİ] Görsel kontrol çıktısı 'ground_truth_kontrol_ornekleri.png' olarak kaydedildi.")
    plt.show()

if __name__ == "__main__":
    siniflar = sinif_isimlerini_yukle(VERI_SETI_DIZINI / "data.yaml")
    poligonlari_kutuya_donustur(VERI_SETI_DIZINI)
    kalite_tablosu = goruntu_kalitesini_tara(VERI_SETI_DIZINI, ALT_KUMELER)
    ornek_etiketleri_cizdir(VERI_SETI_DIZINI, siniflar)