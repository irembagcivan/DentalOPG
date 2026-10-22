import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

VERI_SETI_DIZINI = Path("/Users/irembagcivan/Desktop/DentalOPG/14 class DENTAL.v4i.yolov8")

# Caries (3), Periodontities (10), apical abscss (13), Rootcanel (11) gibi ince bulguları hedefleyelim
HEDEF_SINIFLAR = {
    3: "Caries (Curuk)", 
    10: "Periodontities", 
    13: "Apical Abcess", 
    11: "Rootcanal"
}

def on_isleme_eski(img_gri):
    filtreli = cv2.bilateralFilter(img_gri, d=5, sigmaColor=25, sigmaSpace=25)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kontrastli = clahe.apply(filtreli)
    gauss = cv2.GaussianBlur(kontrastli, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(kontrastli, 1.3, gauss, -0.3, 0)

def on_isleme_yeni_yumusak(img_gri):
    filtreli = cv2.bilateralFilter(img_gri, d=5, sigmaColor=30, sigmaSpace=30)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    kontrastli = clahe.apply(filtreli)
    gauss = cv2.GaussianBlur(kontrastli, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(kontrastli, 1.15, gauss, -0.15, 0)

def rastgele_lezyon_ornekleri_getir(veri_seti_dizini, sinif_sozlugu, her_siniftan_ornek=2):
    goruntu_dizini = veri_seti_dizini / "train" / "images"
    etiket_dizini = veri_seti_dizini / "train" / "labels"
    
    tum_etiketler = list(etiket_dizini.glob("*.txt"))
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
            sinif_id = int(parcalar[0])
            koordinatlar = [float(x) for x in parcalar[1:]]

            if sinif_id in sinif_sozlugu and len(koordinatlar) == 4:
                if len(bulunan_ornekler[sinif_id]) < her_siniftan_ornek:
                    bulunan_ornekler[sinif_id].append((goruntu_yolu, koordinatlar))

    return bulunan_ornekler

def karsilastirma_gorseli_uret():
    bulunan_ornekler = rastgele_lezyon_ornekleri_getir(VERI_SETI_DIZINI, HEDEF_SINIFLAR, her_siniftan_ornek=1)
    
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
        sinif_adi = HEDEF_SINIFLAR[sinif_id]
        for goruntu_yolu, (mx, my, kg, ky) in ornek_listesi:
            img_gri = cv2.imread(str(goruntu_yolu), cv2.IMREAD_GRAYSCALE)
            eski_islenmis = on_isleme_eski(img_gri)
            yeni_islenmis = on_isleme_yeni_yumusak(img_gri)

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
            axes[satir_idx, 0].set_ylabel(sinif_adi, fontsize=10, fontweight='bold')

            satir_idx += 1

    fig.suptitle("İnce Bulgular (Çürük, Apse, Periodontitis) Ön İşleme Karşılaştırması", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    kayit_yolu = VERI_SETI_DIZINI.parent / "lezyon_crop_karsilastirma_duzeltilmis.png"
    plt.savefig(kayit_yolu, dpi=200)
    print(f"[BAŞARILI] Düzenlenmiş karşılaştırma kaydedildi: {kayit_yolu}")
    plt.show()

if __name__ == "__main__":
    karsilastirma_gorseli_uret()