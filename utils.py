import yaml
def sinif_isimlerini_yukle(yaml_yolu):
    """data.yaml dosyasından sınıf isimlerini (names listesi) okur.

    Dosya yoksa ya da 'names' alanı eksikse boş liste döner; script'ler
    bu durumu kendi içinde (ör. sinif_isimleri boşsa raporu atla) ele alır.
    """
    if yaml_yolu.exists():
        with open(yaml_yolu, "r", encoding="utf-8") as dosya:
            veri = yaml.safe_load(dosya)
            return veri.get("names", [])
    return []