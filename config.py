from pathlib import Path

# Veri seti ana dizini (14 class DENTAL.v4i.yolov8)
VERI_SETI_DIZINI = Path("/Users/irembagcivan/Desktop/DentalOPG/14 class DENTAL.v4i.yolov8")

# Boş etiketli dosyaların taşınacağı karantina dizini (03)
KARANTINA_DIZINI = Path("/Users/irembagcivan/Desktop/DentalOPG/karantina_bos_dosyalar")

# Ön işlenmiş görüntülerin yazılacağı dizin (04)
CIKTI_DIZINI = VERI_SETI_DIZINI.with_name(VERI_SETI_DIZINI.name + "_preprocessed")

ALT_KUMELER = ["train", "valid", "test"]
GORUNTU_UZANTILARI = {".jpg", ".jpeg", ".png", ".bmp"}