import cv2

"""
04_gurultu_azaltma_clahe.py ve 05_lezyon_crop_kontrolu.py tarafından paylaşılan
ön işleme fonksiyonu ve parametre setleri.

Önceden bu iki script birbirinden bağımsız iki ayrı ön işleme fonksiyonu
tanımlıyordu; parametre değiştiğinde ikisini elle senkronize etmek
gerekiyordu. Artık tek fonksiyon var, parametre setleri ise birer sözlük
olarak burada tanımlı -> "tek doğruluk kaynağı" (single source of truth).
"""


def goruntu_on_isle(img_gri, bilateral_d, bilateral_sigma_color, bilateral_sigma_space,
                     clahe_clip_limit, clahe_tile_grid, gauss_sigma,
                     keskinlik_agirligi, keskinlik_negatif):
    """Gri görüntüye sırasıyla gürültü azaltma (bilateral filtre),
    CLAHE (lokal kontrast artırma) ve hafif keskinleştirme (unsharp mask) uygular."""
    filtreli = cv2.bilateralFilter(
        img_gri,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    kontrastli = clahe.apply(filtreli)

    gauss_bulanik = cv2.GaussianBlur(kontrastli, (0, 0), sigmaX=gauss_sigma)
    keskinlestirilmis = cv2.addWeighted(kontrastli, keskinlik_agirligi, gauss_bulanik, keskinlik_negatif, 0)

    return keskinlestirilmis


# 04'te veri setinin tamamına uygulanan "resmi" üretim ayarları
URETIM_AYARLARI = dict(
    bilateral_d=5, bilateral_sigma_color=25, bilateral_sigma_space=25,
    clahe_clip_limit=2.0, clahe_tile_grid=(8, 8),
    gauss_sigma=1.0, keskinlik_agirligi=1.3, keskinlik_negatif=-0.3,
)

# 05'te karşılaştırılan "eski" ayar seti (şu an URETIM_AYARLARI ile aynı)
ESKI_AYARLAR = dict(
    bilateral_d=5, bilateral_sigma_color=25, bilateral_sigma_space=25,
    clahe_clip_limit=2.0, clahe_tile_grid=(8, 8),
    gauss_sigma=1.0, keskinlik_agirligi=1.3, keskinlik_negatif=-0.3,
)

# 05'te karşılaştırılan "yeni / dengeli" ayar seti
YENI_YUMUSAK_AYARLAR = dict(
    bilateral_d=5, bilateral_sigma_color=30, bilateral_sigma_space=30,
    clahe_clip_limit=1.5, clahe_tile_grid=(8, 8),
    gauss_sigma=1.2, keskinlik_agirligi=1.15, keskinlik_negatif=-0.15,
)