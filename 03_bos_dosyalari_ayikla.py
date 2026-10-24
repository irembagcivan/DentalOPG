import shutil

from config import VERI_SETI_DIZINI, ALT_KUMELER, KARANTINA_DIZINI


def bos_etiketli_dosyalari_ayikla(veri_seti_dizini, alt_kumeler, karantina_dizini):
    """train/valid/test kümelerindeki boş etiketli (.txt) dosyaları ve eşleşen
    görüntülerini karantina klasörüne taşır. Dosyalar silinmez, gerektiğinde geri alınabilir."""
    (karantina_dizini / "images").mkdir(parents=True, exist_ok=True)
    (karantina_dizini / "labels").mkdir(parents=True, exist_ok=True)

    bos_dosya_sayisi = 0
    print("=== BOŞ ETİKETLİ DOSYALAR ARANIYOR VE TAŞINIYOR ===\n")

    for kume in alt_kumeler:
        etiket_dizini = veri_seti_dizini / kume / "labels"
        goruntu_dizini = veri_seti_dizini / kume / "images"

        if not etiket_dizini.exists():
            print(f"[UYARI] '{kume}/labels' bulunamadı, atlandı.")
            continue

        for etiket_yolu in etiket_dizini.glob("*.txt"):
            with open(etiket_yolu, "r", encoding="utf-8") as dosya:
                satirlar = [s.strip() for s in dosya.readlines() if s.strip()]

            # Eğer dosya tamamen boşsa
            if len(satirlar) == 0:
                bos_dosya_sayisi += 1
                dosya_adi = etiket_yolu.stem

                # Eşleşen görüntünün uzantısını bularak yolunu belirle
                goruntu_yolu = None
                for uzanti in [".jpg", ".jpeg", ".png", ".bmp"]:
                    olasi_yol = goruntu_dizini / f"{dosya_adi}{uzanti}"
                    if olasi_yol.exists():
                        goruntu_yolu = olasi_yol
                        break

                print(f"[{kume.upper()}] Taşınıyor: {dosya_adi}")

                # Etiket dosyasını karantinaya taşı
                shutil.move(str(etiket_yolu), str(karantina_dizini / "labels" / etiket_yolu.name))

                # Görüntü dosyasını karantinaya taşı
                if goruntu_yolu:
                    shutil.move(str(goruntu_yolu), str(karantina_dizini / "images" / goruntu_yolu.name))
                else:
                    print(f"   [UYARI] '{dosya_adi}' için eşleşen görüntü bulunamadı, sadece etiket taşındı.")

    print(f"\n[SONUÇ] Toplam {bos_dosya_sayisi} adet boş etiket ve görüntüsü karantinaya taşındı.")
    print(f"Karantina Dizini: {karantina_dizini}")
    return bos_dosya_sayisi


if __name__ == "__main__":
    bos_etiketli_dosyalari_ayikla(VERI_SETI_DIZINI, ALT_KUMELER, KARANTINA_DIZINI)