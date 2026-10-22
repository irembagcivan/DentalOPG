import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ---------------- Ayarlar ----------------
VERI_SETI_DIZINI = Path("/Users/irembagcivan/Desktop/DentalOPG/14 class DENTAL.v4i.yolov8")
# Çıktı ayrı bir veri seti klasörüne yazılır: <ad>_preprocessed/{train,valid,test}/{images,labels}
CIKTI_DIZINI = VERI_SETI_DIZINI.with_name(VERI_SETI_DIZINI.name + "_preprocessed")
KUMELER = ["train", "valid", "test"]
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp"}

# Gürültü azaltma (ince diş/kök detayını korumak için hafif tutuldu)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 25
BILATERAL_SIGMA_SPACE = 25

# CLAHE
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)

KARSILASTIRMA_ORNEK_SAYISI = 4


# ---------------- 1. Ön işleme ----------------
def goruntu_on_isle(img_gri, clahe):
    """Gri görüntüye önce gürültü azaltma, sonra CLAHE ve hafif keskinleştirme uygular."""
    filtreli = cv2.bilateralFilter(
        img_gri,
        d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE,
    )
    
    kontrastli = clahe.apply(filtreli)
    
    # Unsharp Masking (Keskinleştirme) eklendi
    gauss_bulanik = cv2.GaussianBlur(kontrastli, (0, 0), sigmaX=1.0)
    keskinlestirilmis = cv2.addWeighted(kontrastli, 1.3, gauss_bulanik, -0.3, 0)
    
    return keskinlestirilmis


def goruntulere_on_isleme_uygula(veri_seti_dizini, cikti_dizini):
    print("GÖRÜNTÜLERE GÜRÜLTÜ AZALTMA, CLAHE VE KESKİNLEŞTİRME UYGULANIYOR...")
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID)
    cift_listesi = []
    okunamayan = 0

    for kume in KUMELER:
        orijinal_dizin = veri_seti_dizini / kume / "images"
        if not orijinal_dizin.exists():
            print(f"   Uyarı: {kume}/images bulunamadı, atlandı.")
            continue

        hedef_goruntu_dizini = cikti_dizini / kume / "images"
        hedef_goruntu_dizini.mkdir(parents=True, exist_ok=True)

        dosyalar = sorted(
            p for p in orijinal_dizin.iterdir() if p.suffix.lower() in GORUNTU_UZANTILARI
        )
        print(f"[{kume.upper()}] İşleniyor... ({len(dosyalar)} görüntü)")

        for goruntu_yolu in dosyalar:
            img = cv2.imread(str(goruntu_yolu), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"   Okunamadı: {goruntu_yolu.name}")
                okunamayan += 1
                continue

            iyilestirilmis = goruntu_on_isle(img, clahe)

            # 3 kanallı formata çevir (SAM ve YOLO için uyumlu)
            sonuc_bgr = cv2.cvtColor(iyilestirilmis, cv2.COLOR_GRAY2BGR)

            hedef_yol = hedef_goruntu_dizini / goruntu_yolu.name
            if goruntu_yolu.suffix.lower() in {".jpg", ".jpeg"}:
                # Tekrar sıkıştırmadan kaynaklanan kaliteyi kaybını en aza indir
                cv2.imwrite(str(hedef_yol), sonuc_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            else:
                cv2.imwrite(str(hedef_yol), sonuc_bgr)

            cift_listesi.append((goruntu_yolu, hedef_yol))

        # YOLO, etiketleri görüntü yolundaki "images" -> "labels" değişimiyle bulur;
        # bu yüzden etiketler de yeni veri setine kopyalanır.
        kaynak_etiket = veri_seti_dizini / kume / "labels"
        if kaynak_etiket.exists():
            shutil.copytree(kaynak_etiket, cikti_dizini / kume / "labels", dirs_exist_ok=True)

    # data.yaml dosyasını da yeni veri setine kopyala
    yaml_yolu = veri_seti_dizini / "data.yaml"
    if yaml_yolu.exists():
        shutil.copy2(yaml_yolu, cikti_dizini / "data.yaml")

    if okunamayan:
        print(f"\n   Uyarı: {okunamayan} görüntü okunamadı.")
    print("\n[BAŞARILI] Tüm ön işleme adımları tamamlandı!")
    return cift_listesi


# ---------------- 2. Önizleme ----------------
def karsilastirma_gorseli_olustur(cift_listesi, cikti_dizini, boyut=640):
    """Rastgele örneklerde orijinal | işlenmiş yan yana görseli üretir (sadece önizleme)."""
    if not cift_listesi:
        return

    random.seed(42)
    ornekler = random.sample(cift_listesi, min(KARSILASTIRMA_ORNEK_SAYISI, len(cift_listesi)))

    satirlar = []
    for orijinal_yolu, islenmis_yolu in ornekler:
        a = cv2.imread(str(orijinal_yolu), cv2.IMREAD_COLOR)
        b = cv2.imread(str(islenmis_yolu), cv2.IMREAD_COLOR)
        if a is None or b is None:
            continue
        a = cv2.resize(a, (boyut, boyut))
        b = cv2.resize(b, (boyut, boyut))
        satirlar.append(np.hstack([a, b]))

    if satirlar:
        yol = cikti_dizini / "on_isleme_karsilastirma.png"
        cv2.imwrite(str(yol), np.vstack(satirlar))
        print(f"   Karşılaştırma görseli kaydedildi: {yol}")


if __name__ == "__main__":
    ciftler = goruntulere_on_isleme_uygula(VERI_SETI_DIZINI, CIKTI_DIZINI)
    karsilastirma_gorseli_olustur(ciftler, CIKTI_DIZINI)