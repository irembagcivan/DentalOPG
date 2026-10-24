''' 
Bu script tüm veri setini değil, sadece 4 hedef sınıfa (Caries, Periodontitis, Apical abscess, Rootcanal) ait örnek bölgeleri inceledi. Mantığı:

train kümesinden her sınıf için 1'er örnek görüntü rastgele seçti (artık seed sabit olduğu için her çalıştırmada aynı örnekler çıkıyor).
Etiketli bölgenin (bounding box) etrafına biraz pay (padding) bırakarak o bölgeyi kırptı — yani tüm röntgen yerine sadece lezyonun olduğu küçük 
alanı aldı. Bu, küçük/ince bulguların ön işlemeden nasıl etkilendiğini daha net görmek için.
Her kırpılan bölgeye iki farklı parametre setiyle (Eski: Clip=2.0/Sharp=0.3, Yeni: Clip=1.5/Sharp=0.15) ayrı ayrı ön işleme uyguladı.

Çıkan görsel: lezyon_crop_karsilastirma_duzeltilmis.png
4 satır (her hedef sınıf için 1), her satırda 3 sütun: Orijinal | Eski ayar | Yeni ayar. Sol tarafta sınıf adı etiketli. 
Bu görselin amacı: hangi parametre setinin ince/soluk bulguları (çürük, periodontitis gibi genelde göz le ayırt etmesi zor olan şeyleri) 
daha iyi ortaya çıkardığına karar vermen. README’de de yazdığın gibi, eski ayar daha yüksek kontrast veriyor (soluk bulgularda avantajlı 
ama parlak dolgu/kaplamalarda riskli), yeni ayar daha “doğal” ama farkı gözle net ayırmak zor — bu yüzden nihai kararın model başarısıyla 
(mAP karşılaştırmasıyla) verilmesi gerektiğini zaten belirtmiştin, bu hâlâ geçerli bir plan.

Özetle: 04 tüm veri setini üretime hazırlıyor, 05 ise sadece iki aday parametre setini küçük bir örneklemde gözle karşılaştırman için var
nihai kararı YOLOv8 ile kısa bir deneme eğitimi yapıp mAP’e bakarak vereceksin.
'''

import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import VERI_SETI_DIZINI
from utils import sinif_isimlerini_yukle
from on_isleme import goruntu_on_isle, ESKI_AYARLAR, YENI_YUMUSAK_AYARLAR

# İnce bulgular (küçük/soluk lezyonlar)
HEDEF_SINIF_ADLARI = ["Caries", "Periodontitis", "Apical abscess", "Rootcanal"]


def hedef_siniflari_coz(sinif_isimleri, hedef_adlar):
    """Sınıf adlarını data.yaml sırasına göre ID'lere çözümler.

    Önceden sınıf ID'leri koddaki sözlükte hardcoded idi (data.yaml sırası
    değişirse sessizce yanlış sınıf eşleşirdi). Artık isimden arayarak
    çözümleniyor; bulunamayan bir isim varsa uyarı verip atlanıyor.
    """
    sozluk = {}
    for adi in hedef_adlar:
        eslesen = [i for i, s in enumerate(sinif_isimleri) if s.strip().lower() == adi.lower()]
        if eslesen:
            sozluk[eslesen[0]] = adi
        else:
            print(f"[UYARI] '{adi}' sınıfı data.yaml içinde bulunamadı, atlanıyor.")
    return sozluk


def rastgele_lezyon_ornekleri_getir(veri_seti_dizini, sinif_sozlugu, her_siniftan_ornek=2):
    goruntu_dizini = veri_seti_dizini / "train" / "images"
    etiket_dizini = veri_seti_dizini / "train" / "labels"

    tum_etiketler = list(etiket_dizini.glob("*.txt"))
    random.seed(42)  # Görseller her çalıştırmada aynı örnekleri göstersin (tekrarlanabilirlik)
    random.shuffle(tum_etiketler)

    bulunan_ornekler = {sinif_id: [] for sinif_id in sinif_sozlugu}

    for etiket_yolu in tum_etiketler:
        if all(len(v) >= her_siniftan_ornek for v in bulunan_ornekler.values()):
            break

        goruntu_yolu = goruntu_dizini / f"{etiket_yolu.stem}.jpg"
        if not goruntu_yolu.exists():
            continue

        with open(etiket_yolu, "r", encoding="utf-8") as f:
            satirlar = [s.strip() for s in f.readlines() if s.strip()]

        for satir in satirlar:
            parcalar = satir.split()
            try:
                sinif_id = int(parcalar[0])
                koordinatlar = [float(x) for x in parcalar[1:]]
            except (ValueError, IndexError):
                continue

            if sinif_id in sinif_sozlugu and len(koordinatlar) == 4:
                if len(bulunan_ornekler[sinif_id]) < her_siniftan_ornek:
                    bulunan_ornekler[sinif_id].append((goruntu_yolu, koordinatlar))

    return bulunan_ornekler


def karsilastirma_gorseli_uret():
    sinif_isimleri = sinif_isimlerini_yukle(VERI_SETI_DIZINI / "data.yaml")
    hedef_siniflar = hedef_siniflari_coz(sinif_isimleri, HEDEF_SINIF_ADLARI)

    bulunan_ornekler = rastgele_lezyon_ornekleri_getir(VERI_SETI_DIZINI, hedef_siniflar, her_siniftan_ornek=1)

    aktif_satirlar = [item for sublist in bulunan_ornekler.values() for item in sublist]
    toplam_satir = len(aktif_satirlar)
    if toplam_satir == 0:
        print("Hedef sınıflara ait yeterli örnek bulunamadı.")
        return

    fig, axes = plt.subplots(toplam_satir, 3, figsize=(10, 3.2 * toplam_satir))
    if toplam_satir == 1:
        axes = np.expand_dims(axes, axis=0)

    satir_idx = 0
    for sinif_id, ornek_listesi in bulunan_ornekler.items():
        sinif_adi = hedef_siniflar[sinif_id]
        for goruntu_yolu, (mx, my, kg, ky) in ornek_listesi:
            img_gri = cv2.imread(str(goruntu_yolu), cv2.IMREAD_GRAYSCALE)
            eski_islenmis = goruntu_on_isle(img_gri, **ESKI_AYARLAR)
            yeni_islenmis = goruntu_on_isle(img_gri, **YENI_YUMUSAK_AYARLAR)

            h, w = img_gri.shape
            pad_w, pad_h = kg * 0.8, ky * 0.8
            x1 = max(0, int((mx - kg / 2 - pad_w) * w))
            y1 = max(0, int((my - ky / 2 - pad_h) * h))
            x2 = min(w, int((mx + kg / 2 + pad_w) * w))
            y2 = min(h, int((my + ky / 2 + pad_h) * h))

            crop_orijinal = img_gri[y1:y2, x1:x2]
            crop_eski = eski_islenmis[y1:y2, x1:x2]
            crop_yeni = yeni_islenmis[y1:y2, x1:x2]

            axes[satir_idx, 0].imshow(crop_orijinal, cmap="gray", vmin=0, vmax=255)
            axes[satir_idx, 0].set_title("ORİJİNAL", fontsize=10)
            axes[satir_idx, 0].axis("off")

            axes[satir_idx, 1].imshow(crop_eski, cmap="gray", vmin=0, vmax=255)
            axes[satir_idx, 1].set_title("ESKİ (Clip=2.0, Sharp=0.3)", fontsize=10)
            axes[satir_idx, 1].axis("off")

            axes[satir_idx, 2].imshow(crop_yeni, cmap="gray", vmin=0, vmax=255)
            axes[satir_idx, 2].set_title("YENİ (Clip=1.5, Sharp=0.15)", fontsize=10)
            axes[satir_idx, 2].axis("off")

            # Sol tarafa sınıf adını yazar
            axes[satir_idx, 0].set_ylabel(sinif_adi, fontsize=10, fontweight="bold")

            satir_idx += 1

    fig.suptitle("İnce Bulgular (Çürük, Apse, Periodontitis) Ön İşleme Karşılaştırması", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    kayit_yolu = VERI_SETI_DIZINI.parent / "lezyon_crop_karsilastirma_duzeltilmis.png"
    plt.savefig(kayit_yolu, dpi=200)
    print(f"[BAŞARILI] Düzenlenmiş karşılaştırma kaydedildi: {kayit_yolu}")
    plt.show()


if __name__ == "__main__":
    karsilastirma_gorseli_uret()