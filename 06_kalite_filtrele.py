"""
goruntu_kalite_raporu.csv (02'nin çıktısı) içindeki bulanıklık skorlarına göre
uç (aşırı bulanık / aşırı gürültülü) görüntü adaylarını belirler, gözle kontrol
edebilmen için görsel bir "inceleme grid'i" üretir ve SADECE sen onayladıktan
sonra (KARANTINAYA_TASI = True yaptığında) bu görüntüleri + etiketlerini
silmeden karantina klasörüne taşır.

Kullanım:
  1) KARANTINAYA_TASI = False iken çalıştır -> adayları ve inceleme görsellerini üretir,
     hiçbir dosyayı taşımaz.
  2) 'inceleme_asiri_bulanik.png' ve 'inceleme_asiri_gurultulu.png' dosyalarını
     gözden geçir. Gerçekten kullanılamaz olduğuna karar verdiklerini not al.
     ('dusuk_kalite_adaylari.csv' dosyasını elle düzenleyip tutmak istediğin
     satırları silebilirsin; script bu dosyayı tekrar okumaz, sadece referans içindir.)
  3) Gerekirse config.py'deki BULANIKLIK_ALT_ESIK / BULANIKLIK_UST_ESIK değerlerini ayarla.
  4) Emin olunca KARANTINAYA_TASI = True yapıp tekrar çalıştır.
"""
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    VERI_SETI_DIZINI,
    BULANIKLIK_ALT_ESIK, # 14 ve 17 skorlarını karantinaya almaya karar verdim alt eşiği 18 yaptım.(config.py de)
    BULANIKLIK_UST_ESIK,
    DUSUK_KALITE_KARANTINA_DIZINI,
)

KALITE_RAPORU_CSV = "goruntu_kalite_raporu.csv"
ADAY_LISTESI_CSV = "dusuk_kalite_adaylari.csv"

# Bu bayrak False iken script SADECE inceleme için adayları ve görselleri üretir,
# hiçbir dosyayı taşımaz. Görselleri gözden geçirip emin olduktan sonra True yap.
KARANTINAYA_TASI = True

GRID_SUTUN = 5


def adaylari_belirle(kalite_tablosu):
    """Eşiklerin dışında kalan (aşırı bulanık / aşırı gürültülü) görüntüleri işaretler."""
    tablo = kalite_tablosu.copy()
    tablo["Sebep"] = None
    tablo.loc[tablo["Bulanıklık Skoru"] < BULANIKLIK_ALT_ESIK, "Sebep"] = "ASIRI_BULANIK"
    tablo.loc[tablo["Bulanıklık Skoru"] > BULANIKLIK_UST_ESIK, "Sebep"] = "ASIRI_GURULTULU"
    return tablo[tablo["Sebep"].notna()].sort_values("Bulanıklık Skoru")


def inceleme_grid_olustur(aday_tablosu, sebep, dosya_adi, baslik):
    """Belirtilen sebebe (ASIRI_BULANIK / ASIRI_GURULTULU) ait adayların küçük resimlerini
    tek bir görselde toplar; böylece hepsini tek tek açmadan gözle tarayabilirsin.
    Her küçük resmin altındaki sayı o görüntünün bulanıklık skorudur."""
    alt_tablo = aday_tablosu[aday_tablosu["Sebep"] == sebep]
    if alt_tablo.empty:
        print(f"[BİLGİ] '{sebep}' için aday bulunamadı, grid oluşturulmadı.")
        return

    satir_sayisi = -(-len(alt_tablo) // GRID_SUTUN)  # yukarı yuvarlama
    fig, eksenler = plt.subplots(satir_sayisi, GRID_SUTUN, figsize=(3 * GRID_SUTUN, 3 * satir_sayisi))
    eksenler = np.atleast_1d(eksenler).flatten()

    for ax, (_, satir) in zip(eksenler, alt_tablo.iterrows()):
        goruntu_yolu = VERI_SETI_DIZINI / satir["Alt Küme"] / "images" / satir["Dosya Adı"]
        img = cv2.imread(str(goruntu_yolu), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "okunamadı", ha="center", va="center")
        ax.set_title(f"{satir['Bulanıklık Skoru']:.0f}", fontsize=9)
        ax.axis("off")

    # Kullanılmayan hücreleri gizle
    for ax in eksenler[len(alt_tablo):]:
        ax.axis("off")

    fig.suptitle(baslik, fontsize=13)
    plt.tight_layout()
    plt.savefig(dosya_adi, dpi=150)
    plt.close(fig)
    print(f"[BİLGİ] '{dosya_adi}' kaydedildi ({len(alt_tablo)} görüntü).")


def karantinaya_tasi(aday_tablosu, karantina_dizini):
    """Aday görüntüleri + etiketlerini karantina klasörüne taşır (silmez)."""
    (karantina_dizini / "images").mkdir(parents=True, exist_ok=True)
    (karantina_dizini / "labels").mkdir(parents=True, exist_ok=True)

    tasinan = 0
    for _, satir in aday_tablosu.iterrows():
        kume = satir["Alt Küme"]
        dosya_adi = satir["Dosya Adı"]
        goruntu_yolu = VERI_SETI_DIZINI / kume / "images" / dosya_adi
        etiket_yolu = VERI_SETI_DIZINI / kume / "labels" / f"{Path(dosya_adi).stem}.txt"

        if goruntu_yolu.exists():
            shutil.move(str(goruntu_yolu), str(karantina_dizini / "images" / goruntu_yolu.name))
        if etiket_yolu.exists():
            shutil.move(str(etiket_yolu), str(karantina_dizini / "labels" / etiket_yolu.name))

        tasinan += 1
        print(f"[{kume.upper()}] Karantinaya taşındı: {dosya_adi} "
              f"({satir['Sebep']}, skor={satir['Bulanıklık Skoru']:.1f})")

    print(f"\n[SONUÇ] Toplam {tasinan} görüntü '{karantina_dizini}' klasörüne taşındı.")


if __name__ == "__main__":
    kalite_tablosu = pd.read_csv(KALITE_RAPORU_CSV)
    adaylar = adaylari_belirle(kalite_tablosu)

    print(f"[BİLGİ] Toplam {len(kalite_tablosu)} görüntü tarandı.")
    print(f"[BİLGİ] Aşırı bulanık (skor < {BULANIKLIK_ALT_ESIK}): {(adaylar['Sebep'] == 'ASIRI_BULANIK').sum()}")
    print(f"[BİLGİ] Aşırı gürültülü (skor > {BULANIKLIK_UST_ESIK}): {(adaylar['Sebep'] == 'ASIRI_GURULTULU').sum()}")

    adaylar.to_csv(ADAY_LISTESI_CSV, index=False, encoding="utf-8-sig")
    print(f"[BİLGİ] Aday listesi '{ADAY_LISTESI_CSV}' olarak kaydedildi.")

    inceleme_grid_olustur(adaylar, "ASIRI_BULANIK", "inceleme_asiri_bulanik.png",
                           "Aşırı Bulanık Adaylar (başlıktaki sayı bulanıklık skoru)")
    inceleme_grid_olustur(adaylar, "ASIRI_GURULTULU", "inceleme_asiri_gurultulu.png",
                           "Aşırı Gürültülü Adaylar (başlıktaki sayı bulanıklık skoru)")

    if KARANTINAYA_TASI:
        karantinaya_tasi(adaylar, DUSUK_KALITE_KARANTINA_DIZINI)
    else:
        print("\n[BİLGİ] KARANTINAYA_TASI = False, hiçbir dosya taşınmadı.")
        print("        Görselleri inceledikten sonra emin olursan KARANTINAYA_TASI = True yapıp tekrar çalıştır.")