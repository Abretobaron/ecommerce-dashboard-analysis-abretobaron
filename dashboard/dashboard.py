from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Konfigurasi halaman & gaya visual (disamakan dengan notebook.ipynb)
st.set_page_config(page_title="E-Commerce Dashboard", page_icon="🛒", layout="wide")
sns.set_theme(style="whitegrid")

HIGHLIGHT, MUTED, BAD = "#1f77b4", "#c7c7c7", "#d62728"
DELAY_LABELS = ["Tepat waktu / lebih awal", "Telat 1-3 hari", "Telat 4-7 hari", "Telat 8-14 hari", "Telat >14 hari"]
DELAY_COLORS = dict(zip(DELAY_LABELS, [HIGHLIGHT, MUTED, BAD, BAD, BAD]))
DATA_DIR = Path(__file__).parent


# Load data
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "main_data.csv", parse_dates=["order_purchase_timestamp"])
    geo = pd.read_csv(DATA_DIR / "state_geo.csv")
    return df, geo


# Fungsi analisis (logika sama dengan notebook)
def to_order_level(df):
    """Satu baris per pesanan (data utama berisi satu baris per item)."""
    orders = df.drop_duplicates("order_id").copy()
    orders["delay_group"] = pd.cut(orders["delay_days"], bins=[-np.inf, 0, 3, 7, 14, np.inf], labels=DELAY_LABELS)
    return orders


def category_summary(df):
    summary = (df.groupby("product_category")
               .agg(revenue=("price", "sum"), items_sold=("order_id", "count"), avg_price=("price", "mean"))
               .sort_values("revenue", ascending=False))
    summary["share_pct"] = summary["revenue"] / summary["revenue"].sum() * 100
    return summary


def delay_review_summary(orders):
    return (orders.dropna(subset=["review_score"]).groupby("delay_group", observed=True)
            .agg(orders=("order_id", "count"), avg_review=("review_score", "mean"),
                 pct_score_1_2=("review_score", lambda s: (s <= 2).mean() * 100)))


def state_summary(orders):
    return (orders.groupby("customer_state")
            .agg(orders=("order_id", "count"), late_pct=("is_late", lambda s: s.mean() * 100),
                 avg_delivery_days=("delivery_days", "mean"), avg_review=("review_score", "mean"))
            .sort_values("late_pct", ascending=False))


def rfm_segment(row):
    if row["f_score"] >= 2:
        return "Loyal Aktif" if row["r_score"] >= 3 else "Loyal Berisiko"
    if row["r_score"] >= 3:
        return "Baru Bernilai Tinggi" if row["m_score"] >= 3 else "Baru Bernilai Rendah"
    return "Dormant Bernilai Tinggi" if row["m_score"] >= 3 else "Dormant Bernilai Rendah"


def rfm_summary(df):
    """RFM per pelanggan unik. Skor 1-4 (4 = terbaik), segmen berdasarkan aturan bisnis."""
    reference_date = df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    rfm = (df.groupby("customer_unique_id")
           .agg(last_purchase=("order_purchase_timestamp", "max"), frequency=("order_id", "nunique"),
                monetary=("price", "sum"))
           .reset_index())
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days

    rfm["r_score"] = pd.qcut(rfm["recency"], 4, labels=[4, 3, 2, 1]).astype(int)
    rfm["f_score"] = pd.cut(rfm["frequency"], bins=[0, 1, 2, 3, np.inf], labels=[1, 2, 3, 4]).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm["segment"] = rfm.apply(rfm_segment, axis=1)

    summary = (rfm.groupby("segment")
               .agg(customers=("customer_unique_id", "count"), avg_recency=("recency", "mean"),
                    avg_frequency=("frequency", "mean"), avg_monetary=("monetary", "mean"),
                    total_monetary=("monetary", "sum"))
               .sort_values("customers", ascending=False))
    summary["customer_pct"] = summary["customers"] / summary["customers"].sum() * 100
    summary["revenue_pct"] = summary["total_monetary"] / summary["total_monetary"].sum() * 100
    return summary


# Fungsi visualisasi (sama dengan notebook)
def plot_category(category_df, n):
    top, bottom = category_df.head(n), category_df.tail(n)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(x=top["revenue"] / 1e3, y=top.index, color=HIGHLIGHT, ax=axes[0])
    axes[0].set_title(f"{n} Kategori dengan Revenue Tertinggi", fontsize=14, loc="left")
    sns.barplot(x=bottom["revenue"] / 1e3, y=bottom.index, color=MUTED, ax=axes[1])
    axes[1].set_title(f"{n} Kategori dengan Revenue Terendah", fontsize=14, loc="left")

    for ax, data in zip(axes, [top, bottom]):
        ax.set_xlabel("Revenue (ribu R$)")
        ax.set_ylabel("")
        for i, (rev, share) in enumerate(zip(data["revenue"], data["share_pct"])):
            ax.text(rev / 1e3, i, f"  {rev / 1e3:,.1f}k ({share:.2f}%)", va="center", fontsize=10)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
        sns.despine(ax=ax, left=True)

    fig.suptitle(f"{n} kategori teratas menyumbang {top['share_pct'].sum():.1f}% dari total revenue",
                 fontsize=15, x=0.01, ha="left")
    plt.tight_layout()
    return fig


def plot_delay_review(delay_df):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = delay_df.index.astype(str)
    sns.barplot(x=labels, y=delay_df["avg_review"], palette=[DELAY_COLORS[g] for g in labels],
                hue=labels, legend=False, ax=ax)
    ax.set_title("Semakin lama terlambat, review score semakin anjlok", fontsize=13, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Rata-rata review score (1-5)")
    ax.set_ylim(0, 5)
    ax.tick_params(axis="x", rotation=20)
    for i, v in enumerate(delay_df["avg_review"]):
        ax.text(i, v + 0.07, f"{v:.2f}", ha="center", fontsize=11)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    return fig


def plot_top_late_states(state_df, overall_late, n=10):
    top_late = state_df.head(n).sort_values("late_pct")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(x=top_late["late_pct"], y=top_late.index, color=BAD, ax=ax)
    ax.axvline(overall_late, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{len(top_late)} State dengan Persentase Pesanan Terlambat Tertinggi", fontsize=13, loc="left")
    ax.set_xlabel(f"Pesanan terlambat (%)  |  garis putus-putus = rata-rata {overall_late:.1f}%")
    ax.set_ylabel("")
    for i, v in enumerate(top_late["late_pct"]):
        ax.text(v + 0.2, i, f"{v:.1f}%", va="center", fontsize=10)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    return fig


def plot_rfm(summary):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    order = summary.index
    sns.barplot(x=summary["customer_pct"], y=order, color=MUTED, ax=axes[0])
    axes[0].set_title("Proporsi Pelanggan per Segmen (%)", fontsize=14, loc="left")
    sns.barplot(x=summary["revenue_pct"], y=order, color=HIGHLIGHT, ax=axes[1])
    axes[1].set_title("Kontribusi Revenue per Segmen (%)", fontsize=14, loc="left")
    for ax, col in zip(axes, ["customer_pct", "revenue_pct"]):
        ax.set_xlabel("%")
        ax.set_ylabel("")
        for i, v in enumerate(summary[col]):
            ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=10)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)
        sns.despine(ax=ax, left=True)
    axes[1].set_yticklabels([])
    plt.tight_layout()
    return fig


def plot_geo(geo_df):
    fig, ax = plt.subplots(figsize=(10, 9))
    sc = ax.scatter(geo_df["lng"], geo_df["lat"], s=geo_df["revenue"] / 1500, c=geo_df["late_pct"],
                    cmap="YlOrRd", edgecolor="black", linewidth=0.6, alpha=0.85)
    for _, r in geo_df.iterrows():
        ax.annotate(r["customer_state"], (r["lng"], r["lat"]), ha="center", va="center", fontsize=8)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Pesanan terlambat (%)")
    ax.set_title("Revenue (ukuran lingkaran) vs Keterlambatan (warna) per State di Brasil", fontsize=14, loc="left")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    sns.despine()
    plt.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# Sidebar: filter interaktif
all_data, state_geo = load_data()

st.sidebar.title("🛒 Filter Data")
min_date = all_data["order_purchase_timestamp"].min().date()
max_date = all_data["order_purchase_timestamp"].max().date()
date_range = st.sidebar.date_input("Periode pembelian", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
states = st.sidebar.multiselect("State pelanggan", sorted(all_data["customer_state"].unique()),
                                placeholder="Semua state")
categories = st.sidebar.multiselect("Kategori produk", sorted(all_data["product_category"].unique()),
                                    placeholder="Semua kategori")
top_n = st.sidebar.slider("Jumlah kategori teratas/terbawah", 3, 10, 5)
st.sidebar.caption("Filter berlaku untuk seluruh grafik. Biarkan kosong untuk menampilkan semua data.")

# Terapkan filter
if len(date_range) != 2:
    st.info("Pilih tanggal awal dan akhir pada filter periode.")
    st.stop()
start, end = date_range
mask = all_data["order_purchase_timestamp"].between(
    pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(days=1, seconds=-1))
if states:
    mask &= all_data["customer_state"].isin(states)
if categories:
    mask &= all_data["product_category"].isin(categories)
df = all_data[mask]

if df.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Ubah filter pada sidebar.")
    st.stop()

orders = to_order_level(df)

# Header & KPI
st.title("E-Commerce Public Dataset Dashboard")
st.caption("Pesanan *delivered* periode Januari 2017 - Agustus 2018 | Analisis: kategori produk, keterlambatan "
           "pengiriman, RFM, dan geospasial")

late_pct = orders["is_late"].mean() * 100
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Revenue", f"R$ {df['price'].sum() / 1e6:,.2f} jt")
col2.metric("Jumlah Pesanan", f"{orders['order_id'].nunique():,}")
col3.metric("Pelanggan Unik", f"{df['customer_unique_id'].nunique():,}")
col4.metric("Pesanan Terlambat", f"{late_pct:.2f}%")
col5.metric("Rata-rata Review", f"{orders['review_score'].mean():.2f}")

tab_cat, tab_delay, tab_rfm, tab_geo, tab_conclusion = st.tabs(
    ["📦 Kategori Produk", "🚚 Keterlambatan & Review", "👥 RFM Analysis", "🗺️ Geospasial", "✅ Kesimpulan"])

# Pertanyaan 1: kategori produk
with tab_cat:
    st.subheader("Pertanyaan 1: Kategori produk apa yang menghasilkan revenue tertinggi dan terendah?")
    category_df = category_summary(df)
    show(plot_category(category_df, top_n))

    best = category_df.iloc[0]
    st.markdown(f"**Insight:** Kategori tertinggi adalah **{category_df.index[0]}** "
                f"(R\\$ {best['revenue'] / 1e3:,.1f} ribu, {best['share_pct']:.2f}% dari total revenue). "
                f"{top_n} kategori teratas menyumbang **{category_df.head(top_n)['share_pct'].sum():.1f}%** revenue, "
                f"sedangkan {top_n} kategori terbawah hanya **{category_df.tail(top_n)['share_pct'].sum():.2f}%**.")
    with st.expander("Lihat tabel detail kategori"):
        st.dataframe(category_df.round(2))

# Pertanyaan 2: keterlambatan
with tab_delay:
    st.subheader("Pertanyaan 2: Seberapa besar keterlambatan menurunkan review score, dan state mana yang paling sering terlambat?")
    delay_df = delay_review_summary(orders)
    state_df = state_summary(orders)

    left, right = st.columns(2)
    with left:
        show(plot_delay_review(delay_df))
    with right:
        show(plot_top_late_states(state_df, late_pct))

    review_by_late = orders.groupby("is_late")["review_score"].mean()
    if True in review_by_late.index and False in review_by_late.index:
        st.markdown(f"**Insight:** Rata-rata review pesanan tepat waktu **{review_by_late[False]:.2f}**, "
                    f"sedangkan pesanan terlambat hanya **{review_by_late[True]:.2f}**. "
                    f"State dengan keterlambatan tertinggi: **{state_df.index[0]}** ({state_df['late_pct'].iloc[0]:.1f}%), "
                    f"dibanding rata-rata {late_pct:.1f}%.")
    with st.expander("Lihat tabel detail"):
        st.markdown("**Review score per kelompok keterlambatan**")
        st.dataframe(delay_df.round(2))
        st.markdown("**Ringkasan per state**")
        st.dataframe(state_df.round(2))

# Analisis lanjutan: RFM
with tab_rfm:
    st.subheader("RFM Analysis: segmentasi pelanggan berdasarkan perilaku belanja")
    st.markdown("**Recency** = hari sejak pembelian terakhir, **Frequency** = jumlah pesanan unik, "
                "**Monetary** = total belanja. Skor 1-4 (4 = terbaik); segmen ditentukan dengan aturan bisnis.")
    try:
        rfm_df = rfm_summary(df)
    except ValueError:
        st.warning("Data terlalu sedikit untuk membentuk kuartil RFM. Perluas filter pada sidebar.")
    else:
        show(plot_rfm(rfm_df))
        loyal_pct = rfm_df.loc[rfm_df.index.str.startswith("Loyal"), "customer_pct"].sum()
        high_value = rfm_df.loc[rfm_df.index.str.contains("Tinggi"), "revenue_pct"].sum()
        st.markdown(f"**Insight:** Pelanggan loyal (beli lebih dari sekali) hanya **{loyal_pct:.1f}%** pelanggan, "
                    f"sedangkan segmen bernilai tinggi menyumbang **{high_value:.1f}%** revenue sehingga menjadi "
                    f"target utama program retensi dan reaktivasi.")
        with st.expander("Lihat tabel detail segmen RFM"):
            st.dataframe(rfm_df.round(1))

# Analisis lanjutan: geospasial
with tab_geo:
    st.subheader("Geospatial Analysis: revenue vs keterlambatan per state")
    st.markdown("**Ukuran** lingkaran menunjukkan revenue, **warna** (kuning ke merah) menunjukkan persentase pesanan terlambat.")
    geo_df = (orders.groupby("customer_state")
              .agg(orders=("order_id", "count"), late_pct=("is_late", lambda s: s.mean() * 100)).reset_index()
              .merge(df.groupby("customer_state")["price"].sum().rename("revenue").reset_index(), on="customer_state")
              .merge(state_geo, on="customer_state"))
    left, right = st.columns([3, 2])
    with left:
        show(plot_geo(geo_df))
    with right:
        top_rev = geo_df.sort_values("revenue", ascending=False).head(5).copy()
        top_rev["revenue_share"] = top_rev["revenue"] / geo_df["revenue"].sum() * 100
        st.markdown("**5 state dengan revenue terbesar**")
        st.dataframe(top_rev[["customer_state", "revenue_share", "late_pct"]]
                     .rename(columns={"customer_state": "State", "revenue_share": "Revenue (%)",
                                      "late_pct": "Terlambat (%)"}).round(1),
                     hide_index=True)
        st.markdown("**Insight:** Revenue terpusat di Tenggara (SP, RJ, MG), sedangkan keterlambatan tertinggi berada di "
                    "Timur Laut (AL, MA, PI, CE, SE, BA) yang jauh dari pusat penjual. Wilayah ini berpotensi "
                    "tumbuh jika pengiriman diperbaiki.")


# Kesimpulan & rekomendasi

with tab_conclusion:
    st.subheader("Kesimpulan")
    st.markdown("""
- **Pertanyaan 1:** Revenue didominasi `health_beauty`, `watches_gifts`, `bed_bath_table`, `sports_leisure`, dan
  `computers_accessories` yang bersama-sama menyumbang 39,9% total revenue (R\\$ 13,18 juta). Lima kategori terendah
  masing-masing menyumbang kurang dari 0,01%.
- **Pertanyaan 2:** Keterlambatan menurunkan review dari 4,29 (tepat waktu) menjadi 2,57 (terlambat), dan sekitar 1,7
  bila terlambat lebih dari 7 hari. Keterlambatan tertinggi di AL (24,0%), MA (19,6%), PI (16,0%), CE (15,4%), dan SE (15,4%),
  dibanding rata-rata nasional 8,1%.
- **RFM:** 97,1% pelanggan hanya membeli sekali, dan 78% revenue berasal dari pelanggan bernilai tinggi yang separuhnya dormant.
- **Geospasial:** Revenue terpusat di Tenggara, keterlambatan terparah di Timur Laut.

*Angka di atas berasal dari seluruh data tanpa filter, sesuai notebook.*
""")
    st.subheader("Rekomendasi Action Item")
    st.markdown("""
1. **Prioritaskan stok dan promosi pada 5 kategori teratas**, gunakan bundling/cross-selling, dan evaluasi kategori berkontribusi hampir nol.
2. **Pasang target pengiriman maksimal 3 hari lewat estimasi**, buat peringatan dini untuk pesanan berisiko terlambat, dan beri kompensasi proaktif.
3. **Perbaiki logistik di AL, MA, PI, CE, SE, dan BA** (mitra kurir regional, cross-docking, estimasi realistis), serta prioritaskan RJ.
4. **Jalankan kampanye reaktivasi untuk segmen Dormant Bernilai Tinggi** dan insentif pembelian kedua untuk segmen Baru Bernilai Tinggi.
""")


