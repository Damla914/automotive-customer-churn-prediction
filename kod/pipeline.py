import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore") #Uyarılar gösterilmez

CUTOFF = pd.Timestamp("2025-06-30") #Özelliklerin hesaplandığı kullanılacak geçmiş veriler için kesim tarihi.
TARGET_END = pd.Timestamp("2026-06-30") #Churn hedef dönemi, kesim tarihinden sonraki 12 ay olarak belirlenmiştir.

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "veri"
DEFAULT_OUTPUT = ROOT / "ciktilar"


# ---------------------------------------------------------
# Tarih işlemleri
# ---------------------------------------------------------

def parse_date(value):
    """
    Tarihleri biçimlerine göre dönüştürür.
    ISO: yyyy-mm-dd
    Eğik çizgi: gg/aa/yyyy
    Nokta: gg.aa.yyyy
    """
    if pd.isna(value):    # Eksik değerler için NaT döndür
        return pd.NaT

    value = str(value).strip()  # değerleri metin olarak alır ve baştaki ve sondaki boşlukları temizler

    if not value:  # Boş metinler için NaT döndür
        return pd.NaT

    if re.match(r"^\d{4}-\d{2}-\d{2}", value):   # 2025-06-30 gibi değerleri döndürür
        return pd.to_datetime(
            value[:10],
            format="%Y-%m-%d",
            errors="coerce"
        )

    if "/" in value:   # gg/aa/yyyy biçimindeki değerleri döndürür
        return pd.to_datetime(
            value,
            format="%d/%m/%Y",
            errors="coerce"
        )

    if "." in value:   # gg.aa.yyyy biçimindeki değerleri döndürür
        return pd.to_datetime(
            value,
            format="%d.%m.%Y",
            errors="coerce"
        )

    return pd.NaT     # Diğer biçimler için NaT döndür


def slash_dot_check(series):
    """
    '/' ve '.' ayraçlarıyla yazılmış tarihleri inceler.
    İlk ve ikinci parçanın 12'den büyük olma sayılarını
    göstererek gg/aa/yyyy varsayımını kontrol etmeye yardımcı olur.

    Bu fonksiyon tarihleri değiştirmez.
    """
    values = series.dropna().astype(str).str.strip()

    for sep in ("/", "."):
        mask = (
            values.str.contains(re.escape(sep), regex=True)
            & ~values.str.match(r"^\d{4}")
        )
        selected = values[mask]

        first_parts = []
        second_parts = []

        for value in selected:
            parts = value.split(sep)

            if len(parts) < 2:
                continue

            try:
                first_parts.append(int(parts[0]))
                second_parts.append(int(parts[1]))
            except ValueError:
                continue

        first = pd.Series(first_parts, dtype="float64")
        second = pd.Series(second_parts, dtype="float64")

        print(
            f"'{sep}' biçimi: n={len(first)}, "
            f"1. parça > 12: {(first > 12).sum()}, "
            f"2. parça > 12: {(second > 12).sum()}"
        )


# ---------------------------------------------------------
# Şehir standardizasyonu
# ---------------------------------------------------------

TR = str.maketrans({
    "İ": "i",
    "I": "i",
    "ı": "i"
})


def norm_city(value):     # şehir adlarını normalize eder. Örn: "İstanbul" -> "istanbul"
    if pd.isna(value):
        return np.nan

    return str(value).translate(TR).lower().strip()


# ---------------------------------------------------------
# Veri yükleme ve temizlik
# ---------------------------------------------------------

def load_clean(veri):
    m = pd.read_csv(veri / "musteriler.csv")
    a = pd.read_csv(veri / "araclar.csv")
    s = pd.read_csv(veri / "servis_kayitlari.csv")
    g = pd.read_csv(veri / "geri_bildirimler.csv")

    print("İlk kayıt sayıları:")
    print("Müşteriler:", len(m))
    print("Araçlar:", len(a))
    print("Servis kayıtları:", len(s))
    print("Geri bildirimler:", len(g))

    # Tekrarlanan kayıtları kaldır
    m = m.drop_duplicates().copy()
    a = a.drop_duplicates().copy()
    s = s.drop_duplicates().copy()

    print("\nTekrarlar kaldırıldıktan sonra:")
    print("Müşteriler:", len(m))
    print("Araçlar:", len(a))
    print("Servis kayıtları:", len(s))

    # Tarih biçimlerini parse işleminden önce incele
    print("\nServis tarihi biçimi doğrulaması:")
    slash_dot_check(s["servis_tarihi"])

    # Tarih sütunlarını dönüştür
    m["ilk_temas_tarihi"] = m["ilk_temas_tarihi"].map(parse_date)

    for col in ("satin_alma_tarihi", "garanti_bitis_tarihi"):
        a[col] = a[col].map(parse_date)

    s["servis_tarihi"] = s["servis_tarihi"].map(parse_date)
    g["tarih"] = g["tarih"].map(parse_date)

    # Şehir adlarını standardize et
    canonical_cities = {
        "istanbul": "İstanbul",
        "ankara": "Ankara",
        "izmir": "İzmir",
        "bursa": "Bursa",
        "antalya": "Antalya",
        "kocaeli": "Kocaeli",
        "adana": "Adana"
    }

    m["sehir"] = m["sehir"].map(
        lambda x: canonical_cities.get(norm_city(x), x)
        if pd.notna(x) else np.nan
    )

    # Gerçekçi olmayan doğum yıllarını eksik değer yap
    age = m["ilk_temas_tarihi"].dt.year - m["dogum_yili"]

    invalid_birth_year = (
        (m["dogum_yili"] < 1920)
        | (age < 18)
        | (m["musteri_tipi"] == "Kurumsal")
    )

    m.loc[invalid_birth_year, "dogum_yili"] = np.nan

    # Negatif kilometre değerlerini eksik değer yap
    s.loc[s["km"] < 0, "km"] = np.nan

    # Tutarların mutlak değerini al.
    # Bu işlem, negatif tutarların veri girişindeki işaret hatası
    # olduğu varsayımına dayanır; iade/indirim varsa doğrulanmalıdır.
    s["tutar"] = s["tutar"].abs()

    return m, a, s, g


# ---------------------------------------------------------
# Müşteri bazlı veri seti oluşturma
# ---------------------------------------------------------

def build_dataset(m, a, s, cutoff, target_end, use_leaky=False):

    # Kesim tarihinden sonra alınan araçlar analize dahil edilmez
    a = a[a["satin_alma_tarihi"] <= cutoff].copy()

    # Servis kayıtlarını araçlarla eşleştir
    s = s.merge(
        a[[
            "arac_id",
            "musteri_id",
            "model_yili",
            "garanti_bitis_tarihi"
        ]],
        on="arac_id",
        how="inner",
        validate="many_to_one"
    )

    # Kesim tarihine kadar olan geçmiş servisler
    past = s[s["servis_tarihi"] <= cutoff].copy()

    # Hedef dönemdeki servisler
    future = s[
        (s["servis_tarihi"] > cutoff)
        & (s["servis_tarihi"] <= target_end)
    ]

    # Uç tutarları %99 yüzdelik değerde sınırla
    if not past.empty:
        cap = past["tutar"].quantile(0.99)
        past["tutar"] = past["tutar"].clip(upper=cap)

    # Kesim tarihinde geçmiş servis kaydı olan müşteriler
    population = past["musteri_id"].unique()

    # Müşteri bazlı gruplama
    grouped = past.sort_values("servis_tarihi").groupby("musteri_id")

    features = pd.DataFrame({
        "frequency": grouped.size(),
        "monetary": grouped["tutar"].sum(),
        "ort_tutar": grouped["tutar"].mean(),

        "recency_gun": (
            cutoff - grouped["servis_tarihi"].max()
        ).dt.days,

        "ilk_servis_gecen_gun": (
            cutoff - grouped["servis_tarihi"].min()
        ).dt.days,

        "ort_memnuniyet": grouped["memnuniyet_puani"].mean(),
        "son_memnuniyet": grouped["memnuniyet_puani"].last(),

        "anket_cevap_orani": grouped["memnuniyet_puani"].apply(
            lambda x: x.notna().mean()
        ),

        "dusuk_puan_orani": grouped["memnuniyet_puani"].apply(
            lambda x: (x <= 2).sum() / max(x.notna().sum(), 1)
        ),

        "ariza_orani": grouped["servis_tipi"].apply(
            lambda x: (x == "Arıza Onarım").mean()
        ),

        "ort_aralik_gun": grouped["servis_tarihi"].apply(
            lambda x: x.diff().dt.days.mean()
        )
    })

    # Son servis tarihinin ortalama servis aralığına oranı
    features["recency_aralik_orani"] = (
        features["recency_gun"]
        / features["ort_aralik_gun"].replace(0, np.nan)
    )

    # Son 12 aydaki servis sayısı
    last_12_months = past[
        past["servis_tarihi"] > cutoff - pd.Timedelta(days=365)
    ].groupby("musteri_id").size()

    features["ziyaret_son12ay"] = (
        last_12_months.reindex(features.index).fillna(0)
    )

    features["ziyaret_onceki"] = (
        features["frequency"] - features["ziyaret_son12ay"]
    )

    # Araç bazlı özellikleri müşteri düzeyine getir
    vehicle_features = a.groupby("musteri_id").agg(
        arac_sayisi=("arac_id", "count"),
        ort_model_yili=("model_yili", "mean"),
        garanti_bitis=("garanti_bitis_tarihi", "max")
    )

    features = features.join(vehicle_features)

    features["garanti_bitti_mi"] = (
        features["garanti_bitis"] < cutoff
    ).astype(int)

    features["garanti_bitise_gun"] = (
        features["garanti_bitis"] - cutoff
    ).dt.days

    features = features.drop(
        columns=["garanti_bitis"]
    ).reset_index()

    # Müşteri özellikleri
    customer_features = m.assign(
        kidem_gun=(cutoff - m["ilk_temas_tarihi"]).dt.days,
        yas=cutoff.year - m["dogum_yili"]
    )[[
        "musteri_id",
        "musteri_tipi",
        "kayit_kanali",
        "sehir",
        "kidem_gun",
        "yas"
    ]]

    features = features.merge(
        customer_features,
        on="musteri_id",
        how="left",
        validate="one_to_one"
    )

    # Yalnızca veri sızıntısını göstermek için kullanılır.
    # Bu alanlar gerçek modelde kullanılmamalıdır.
    if use_leaky:
        features = features.merge(
            m[[
                "musteri_id",
                "crm_musteri_durumu",
                "son_iletisim_notu"
            ]],
            on="musteri_id",
            how="left",
            validate="one_to_one"
        )

    features = features[
        features["musteri_id"].isin(population)
    ].copy()

    # Hedef dönemde servis kaydı olmayan müşteri churn kabul edilir
    future_customers = future["musteri_id"].unique()

    features["churn"] = (
        ~features["musteri_id"].isin(future_customers)
    ).astype(int)

    return features


# ---------------------------------------------------------
# Model oluşturma
# ---------------------------------------------------------

def make_model(kind, X):

    categorical = [
        col for col in X.columns
        if X[col].dtype == object
    ]

    numeric = [
        col for col in X.columns
        if col not in categorical
    ]

    if kind == "baseline":
        return DummyClassifier(strategy="prior")

    if kind == "logreg":
        preprocessor = ColumnTransformer([
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True
                    ),
                    StandardScaler()
                ),
                numeric
            ),
            (
                "categorical",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="ignore")
                ),
                categorical
            )
        ])

        return make_pipeline(
            preprocessor,
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )

    preprocessor = ColumnTransformer([
        (
            "numeric",
            SimpleImputer(strategy="median"),
            numeric
        ),
        (
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            ),
            categorical
        )
    ])

    return make_pipeline(
        preprocessor,
        HistGradientBoostingClassifier(
            max_depth=3,
            learning_rate=0.05,
            max_iter=200,
            random_state=0
        )
    )


# ---------------------------------------------------------
# Model değerlendirme metrikleri
# ---------------------------------------------------------

def metrics(y, probabilities):

    y = pd.Series(y).reset_index(drop=True)
    probabilities = np.asarray(probabilities)

    top_count = max(1, int(np.ceil(0.10 * len(y))))
    top_indices = np.argsort(-probabilities)[:top_count]

    churn_rate = y.mean()

    return {
        "ROC_AUC": roc_auc_score(y, probabilities),
        "PR_AUC": average_precision_score(y, probabilities),
        "recall@top10%": (
            y.iloc[top_indices].sum() / max(y.sum(), 1)
        ),
        "lift@top10%": (
            y.iloc[top_indices].mean() / churn_rate
            if churn_rate > 0 else np.nan
        )
    }


def run_models(df, output_dir, label):

    X = df.drop(columns=["musteri_id", "churn"]).copy()
    y = df["churn"].reset_index(drop=True)

    for col in X.select_dtypes(
        include=["string", "category"]
    ).columns:
        X[col] = X[col].astype(object)

    results = {}

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    for model_name in ("baseline", "logreg", "gbm"):

        model = make_model(model_name, X)

        probabilities = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            method="predict_proba"
        )[:, 1]

        results[model_name] = metrics(y, probabilities)

    result_df = pd.DataFrame(results).T.round(3)

    print(
        f"\n[{label}] 5-kat CV "
        f"(churn oranı={y.mean():.3f})\n",
        result_df
    )

    result_df.to_csv(
        output_dir / f"metrikler_{label}.csv"
    )

    return result_df


# ---------------------------------------------------------
# Ana analiz akışı
# ---------------------------------------------------------

def main(veri, cikti):

    veri = Path(veri)
    cikti = Path(cikti)
    cikti.mkdir(parents=True, exist_ok=True)

    m, a, s, g = load_clean(veri)

    print(
        "\nKesim sonrası servis:",
        (s["servis_tarihi"] > CUTOFF).sum(),
        "| hedef pencere sonrası:",
        (s["servis_tarihi"] > TARGET_END).sum()
    )

    # Ana müşteri veri seti
    df = build_dataset(
        m, a, s,
        CUTOFF,
        TARGET_END
    )

    print(
        "\nPopülasyon:",
        len(df),
        "| churn dağılımı:\n",
        df["churn"].value_counts()
        .to_frame()
        .assign(oran=lambda x: x / x.sum())
    )

    df.to_csv(
        cikti / "analiz_tablosu.csv",
        index=False
    )

    # Çapraz doğrulama
    run_models(df, cikti, "cv_ana")

    # Zaman bazlı doğrulama
    train_df = build_dataset(
        m, a, s,
        pd.Timestamp("2024-06-30"),
        pd.Timestamp("2025-06-30")
    )

    X_train = train_df.drop(
        columns=["musteri_id", "churn"]
    ).copy()
    y_train = train_df["churn"]

    X_test = df.drop(
        columns=["musteri_id", "churn"]
    ).copy()
    y_test = df["churn"]

    for X_part in (X_train, X_test):
        for col in X_part.select_dtypes(
            include=["string", "category"]
        ).columns:
            X_part[col] = X_part[col].astype(object)

    time_model = make_model("gbm", X_train)
    time_model.fit(X_train, y_train)

    time_probabilities = time_model.predict_proba(X_test)[:, 1]

    print(
        "\nZaman bazlı test:",
        {
            key: round(value, 3)
            for key, value in metrics(
                y_test,
                time_probabilities
            ).items()
        }
    )

    # Veri sızıntısı gösterimi
    leaky_df = build_dataset(
        m, a, s,
        CUTOFF,
        TARGET_END,
        use_leaky=True
    )

    run_models(
        leaky_df,
        cikti,
        "cv_SIZINTILI_crm_dahil"
    )

    print("\nCRM durumu ve churn:")
    print(
        pd.crosstab(
            leaky_df["crm_musteri_durumu"],
            leaky_df["churn"],
            normalize="index"
        ).round(3)
    )

    print("\nSon iletişim notu ve churn:")
    print(
        pd.crosstab(
            leaky_df["son_iletisim_notu"].fillna("yok"),
            leaky_df["churn"],
            normalize="index"
        ).round(3)
    )

    # Değişken önemi
    from sklearn.inspection import permutation_importance

    X = df.drop(
        columns=["musteri_id", "churn"]
    ).copy()
    y = df["churn"]

    for col in X.select_dtypes(
        include=["string", "category"]
    ).columns:
        X[col] = X[col].astype(object)

    importance_model = make_model("gbm", X)
    importance_model.fit(X, y)

    permutation = permutation_importance(
        importance_model,
        X,
        y,
        scoring="average_precision",
        n_repeats=5,
        random_state=0
    )

    importance = pd.Series(
        permutation.importances_mean,
        index=X.columns
    ).sort_values(ascending=False)

    importance.to_csv(
        cikti / "degisken_onemi.csv"
    )

    print(
        "\nİlk 5 değişken:\n",
        importance.head(5).round(4)
    )

    # Servis merkezi memnuniyet analizi
    rated = s.dropna(
        subset=["memnuniyet_puani"]
    ).copy()

    summary = rated.groupby(
        "servis_merkezi"
    )["memnuniyet_puani"].agg(
        ["count", "mean", "std"]
    )

    summary["ci95"] = (
        1.96 * summary["std"]
        / np.sqrt(summary["count"])
    )

    summary["cevap_orani"] = (
        s.groupby("servis_merkezi")["memnuniyet_puani"]
        .apply(lambda x: x.notna().mean())
    )

    summary["islem_sayisi"] = (
        s.groupby("servis_merkezi").size()
    )

    service_type_mean = rated.groupby(
        "servis_tipi"
    )["memnuniyet_puani"].transform("mean")

    summary["tur_duzeltmeli_fark"] = (
        rated["memnuniyet_puani"] - service_type_mean
    ).groupby(rated["servis_merkezi"]).mean()

    summary = summary.sort_values("mean").round(3)

    print("\nServis merkezi memnuniyet özeti:")
    print(summary)

    summary.to_csv(
        cikti / "servis_merkezi_memnuniyet.csv"
    )

    # Servis merkezleri arasında Kruskal-Wallis testi
    groups = [
        group["memnuniyet_puani"].values
        for _, group in rated.groupby("servis_merkezi")
        if len(group) > 0
    ]

    if len(groups) >= 2:
        H, p_value = stats.kruskal(*groups)
        epsilon_squared = H / (len(rated) - 1)

        print(
            f"Kruskal-Wallis: H={H:.2f}, "
            f"p={p_value:.4f}, "
            f"epsilon^2={epsilon_squared:.5f}"
        )

    # Müşteri-merkez düzeyinde tekrar test
    customer_center = (
        rated.merge(
            a[["arac_id", "musteri_id"]],
            on="arac_id",
            how="inner"
        )
        .groupby(
            ["musteri_id", "servis_merkezi"]
        )["memnuniyet_puani"]
        .mean()
        .reset_index()
    )

    customer_groups = [
        group["memnuniyet_puani"].values
        for _, group in customer_center.groupby("servis_merkezi")
        if len(group) > 0
    ]

    if len(customer_groups) >= 2:
        H2, p2 = stats.kruskal(*customer_groups)

        print(
            f"Müşteri-merkez düzeyi Kruskal: "
            f"H={H2:.2f}, p={p2:.4f}"
        )

    # Grafik
    if not summary.empty:
        plt.figure(figsize=(7, 4))
        summary["mean"].plot.barh(
            xerr=summary["ci95"]
        )
        plt.xlim(3.5, 4.3)
        plt.xlabel("Ortalama memnuniyet (%95 GA)")
        plt.tight_layout()
        plt.savefig(
            cikti / "merkez_memnuniyet.png",
            dpi=130
        )
        plt.close()

    # Memnuniyet ile churn ilişkisi
    customer_ratings = df.dropna(
        subset=["ort_memnuniyet"]
    )

    churn_no = customer_ratings.loc[
        customer_ratings["churn"] == 0,
        "ort_memnuniyet"
    ]

    churn_yes = customer_ratings.loc[
        customer_ratings["churn"] == 1,
        "ort_memnuniyet"
    ]

    if len(churn_no) > 0 and len(churn_yes) > 0:
        mw = stats.mannwhitneyu(
            churn_no,
            churn_yes,
            alternative="two-sided"
        )

        print(
            f"\nMemnuniyet: churn=0 {churn_no.mean():.2f} "
            f"| churn=1 {churn_yes.mean():.2f} "
            f"| Mann-Whitney p={mw.pvalue:.2e}"
        )

    print(
        "Anket cevap oranı churn'e göre:\n",
        df.groupby("churn")["anket_cevap_orani"]
        .mean()
        .round(3)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--veri",
        default=str(DEFAULT_DATA)
    )

    parser.add_argument(
        "--cikti",
        default=str(DEFAULT_OUTPUT)
    )

    args = parser.parse_args()

    main(args.veri, args.cikti)
