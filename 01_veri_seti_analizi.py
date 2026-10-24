'''Aşağıdaki betik; train, valid ve test klasörlerini tarayarak görüntü-etiket eşleşmelerini kontrol eder,
her bir .txt dosyasını hem dosya bazında hem de satır (nesne) bazında analiz ederek kaç tanesinin sadece Bounding Box,
sadece Poligon, Karışık (Mixed) veya Boş olduğunu raporlar'''

import pandas as pd

from config import VERI_SETI_DIZINI, ALT_KUMELER, GORUNTU_UZANTILARI
from utils import sinif_isimlerini_yukle


def veri_setini_analiz_et(veri_seti_dizini, alt_kumeler):
    """Görüntü-etiket eşleşmesini ve etiket formatlarını (Kutu vs. Poligon) analiz eder."""
    sinif_isimleri = sinif_isimlerini_yukle(veri_seti_dizini / "data.yaml")
    ozet_kayitlar = []

    # Sınıf bazlı istatistik sözlüğü
    sinif_istatistikleri = {
        i: {"sinirlayici_kutu": 0, "poligon": 0}
        for i in range(len(sinif_isimleri) or 20)
    }

    for kume in alt_kumeler:
        goruntu_dizini = veri_seti_dizini / kume / "images"
        etiket_dizini = veri_seti_dizini / kume / "labels"

        if not goruntu_dizini.exists() or not etiket_dizini.exists():
            print(f"[UYARI] '{kume}' klasöründe 'images' veya 'labels' dizini bulunamadı!")
            continue

        # Dosya isimlerini uzantısız (stem) olarak sözlükte toplama
        goruntu_dosyalari = {
            p.stem: p for p in goruntu_dizini.iterdir()
            if p.suffix.lower() in GORUNTU_UZANTILARI
        }
        etiket_dosyalari = {
            p.stem: p for p in etiket_dizini.iterdir()
            if p.suffix.lower() == ".txt"
        }

        goruntu_isimleri = set(goruntu_dosyalari.keys())
        etiket_isimleri = set(etiket_dosyalari.keys())

        eslesen_isimler = goruntu_isimleri.intersection(etiket_isimleri)
        etiketsiz_goruntuler = goruntu_isimleri - etiket_isimleri
        goruntusuz_etiketler = etiket_isimleri - goruntu_isimleri

        # Dosya bazlı sayaçlar
        bos_dosya_sayisi = 0
        sadece_kutu_dosya_sayisi = 0
        sadece_poligon_dosya_sayisi = 0
        karisik_dosya_sayisi = 0
        bozuk_dosya_sayisi = 0

        # Satır (nesne/etiket) bazlı sayaçlar
        toplam_kutu_nesnesi = 0
        toplam_poligon_nesnesi = 0
        bozuk_satir_sayisi = 0

        for dosya_adi in eslesen_isimler:
            etiket_yolu = etiket_dosyalari[dosya_adi]
            with open(etiket_yolu, "r", encoding="utf-8") as dosya:
                satirlar = [satir.strip() for satir in dosya.readlines() if satir.strip()]

            # Etiket dosyası tamamen boşsa
            if len(satirlar) == 0:
                bos_dosya_sayisi += 1
                continue

            kutu_var_mi = False
            poligon_var_mi = False
            hata_var_mi = False

            for satir in satirlar:
                parcalar = satir.split()

                # Satır, sınıf ID'si ve sayısal koordinatlardan oluşmuyorsa
                # (örn. bozuk/eksik yazılmış bir satırsa) script'i çökertmeden atla.
                try:
                    sinif_id = int(parcalar[0])
                    koordinat_sayisi = len(parcalar) - 1
                    [float(x) for x in parcalar[1:]]  # sayısal olduğunu doğrula
                except (ValueError, IndexError):
                    hata_var_mi = True
                    bozuk_satir_sayisi += 1
                    print(f"[UYARI] Bozuk satır atlandı -> {etiket_yolu.name}: '{satir}'")
                    continue

                # YOLOv8 Sınırlayıcı Kutu: sinif_id + 4 koordinat (merkez_x, merkez_y, genislik, yukseklik)
                if koordinat_sayisi == 4:
                    kutu_var_mi = True
                    toplam_kutu_nesnesi += 1
                    if sinif_id in sinif_istatistikleri:
                        sinif_istatistikleri[sinif_id]["sinirlayici_kutu"] += 1

                # YOLOv8 Poligon: sinif_id + en az 3 köşe noktası (6 koordinat ve çift sayı)
                elif koordinat_sayisi >= 6 and koordinat_sayisi % 2 == 0:
                    poligon_var_mi = True
                    toplam_poligon_nesnesi += 1
                    if sinif_id in sinif_istatistikleri:
                        sinif_istatistikleri[sinif_id]["poligon"] += 1
                else:
                    hata_var_mi = True

            if hata_var_mi:
                bozuk_dosya_sayisi += 1
            elif kutu_var_mi and poligon_var_mi:
                karisik_dosya_sayisi += 1
            elif kutu_var_mi:
                sadece_kutu_dosya_sayisi += 1
            elif poligon_var_mi:
                sadece_poligon_dosya_sayisi += 1

        ozet_kayitlar.append({
            "Alt Küme": kume,
            "Toplam Görüntü": len(goruntu_isimleri),
            "Toplam Etiket (.txt)": len(etiket_isimleri),
            "Eşleşen Çift": len(eslesen_isimler),
            "Etiketsiz Görüntü": len(etiketsiz_goruntuler),
            "Görüntüsüz Etiket": len(goruntusuz_etiketler),
            "Boş Etiket Dosyası": bos_dosya_sayisi,
            "Sadece Kutu (Dosya)": sadece_kutu_dosya_sayisi,
            "Sadece Poligon (Dosya)": sadece_poligon_dosya_sayisi,
            "Karışık Format (Dosya)": karisik_dosya_sayisi,
            "Bozuk Format (Dosya)": bozuk_dosya_sayisi,
            "Bozuk Satır Sayısı": bozuk_satir_sayisi,
            "Toplam Kutu Nesnesi": toplam_kutu_nesnesi,
            "Toplam Poligon Nesnesi": toplam_poligon_nesnesi
        })

    ozet_tablo = pd.DataFrame(ozet_kayitlar)
    print("\n=== VERİ SETİ GENEL DURUM VE FORMAT RAPORU ===")
    print(ozet_tablo.to_string(index=False))

    if sinif_isimleri:
        print("\n=== SINIF BAZLI ETİKET FORMATI DAĞILIMI (TÜM KÜMELER) ===")
        sinif_kayitlari = []
        for sinif_id, sinif_adi in enumerate(sinif_isimleri):
            sinif_kayitlari.append({
                "Sınıf ID": sinif_id,
                "Sınıf Adı": sinif_adi,
                "Sınırlayıcı Kutu Sayısı": sinif_istatistikleri[sinif_id]["sinirlayici_kutu"],
                "Poligon Maske Sayısı": sinif_istatistikleri[sinif_id]["poligon"]
            })
        sinif_tablosu = pd.DataFrame(sinif_kayitlari)
        print(sinif_tablosu.to_string(index=False))


if __name__ == "__main__":
    veri_setini_analiz_et(VERI_SETI_DIZINI, ALT_KUMELER)