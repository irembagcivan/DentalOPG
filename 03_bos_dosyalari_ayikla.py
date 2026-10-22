import shutil
from pathlib import Path

# Dizin yolları
VERI_SETI_DIZINI = Path("/Users/irembagcivan/Desktop/DentalOPG/14 class DENTAL.v4i.yolov8")
ALT_KUMELER = ["train", "valid", "test"]
KARANTINA_DIZINI = Path("/Users/irembagcivan/Desktop/DentalOPG/karantina_bos_dosyalar")

# Karantina alt klasörlerini oluştur
(KARANTINA_DIZINI / "images").mkdir(parents=True, exist_ok=True)
(KARANTINA_DIZINI / "labels").mkdir(parents=True, exist_ok=True)

bos_dosya_sayisi = 0

print("=== BOŞ ETİKETLİ DOSYALAR ARANIYOR VE TAŞINIYOR ===\n")

for kume in ALT_KUMELER:
    etiket_dizini = VERI_SETI_DIZINI / kume / "labels"
    goruntu_dizini = VERI_SETI_DIZINI / kume / "images"
    
    if not etiket_dizini.exists():
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
            shutil.move(str(etiket_yolu), str(KARANTINA_DIZINI / "labels" / etiket_yolu.name))
            
            # Görüntü dosyasını karantinaya taşı
            if goruntu_yolu:
                shutil.move(str(goruntu_yolu), str(KARANTINA_DIZINI / "images" / goruntu_yolu.name))

print(f"\n[SONUÇ] Toplam {bos_dosya_sayisi} adet boş etiket ve görüntüsü karantinaya taşındı.")
print(f"Karantina Dizini: {KARANTINA_DIZINI}")