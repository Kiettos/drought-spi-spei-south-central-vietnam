"""
app.py
======
Giao diện Web App dự báo hạn hán và quản lý Pipeline (Có thanh tiến trình).
Chạy: streamlit run app.py

Dự báo nhiều bước:
    t+1 : dùng đặc trưng quan trắc thực tế            -> độ tin cậy CAO
    t+2 : mọi đặc trưng trễ đều xác định chính xác,    -> TRUNG BÌNH
          chỉ các biến nền của tháng đích dùng khí hậu trung bình
    t+3 : thêm một tháng phải giả định                 -> THẤP
"""
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import xarray as xr
import xgboost as xgb

st.set_page_config(
    page_title="Hệ thống Dự báo Hạn hán",
    layout="wide",
    page_icon="🌵",
    initial_sidebar_state="expanded",
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("outputs/models")
INTERIM_NC = Path("data/interim/era5_processed_nam_trung_bo.nc")
INDICES_NC = Path("data/processed/indices_nam_trung_bo.nc")

INDEX_COLS = ["SPI_1", "SPI_3", "SPI_6", "SPI_12",
              "SPEI_1", "SPEI_3", "SPEI_6", "SPEI_12"]
BASE_COLS = ["tp", "t2m"] + INDEX_COLS

CONFIDENCE = {
    1: dict(level="high", icon="🟢", label="Độ tin cậy CAO", box="success",
            note="Dùng toàn bộ đặc trưng quan trắc thực tế của tháng đầu vào. "
                 "Đây là dự báo một bước, đúng điều kiện huấn luyện."),
    2: dict(level="medium", icon="🟡", label="Độ tin cậy TRUNG BÌNH", box="warning",
            note="Các đặc trưng trễ (mưa, nhiệt độ, SPEI-3 các tháng trước) vẫn xác định "
                 "chính xác từ số liệu đã quan trắc, nhưng lượng mưa và nhiệt độ của chính "
                 "tháng đích chưa biết nên dùng giá trị khí hậu trung bình nhiều năm."),
    3: dict(level="low", icon="🔴", label="Độ tin cậy THẤP", box="error",
            note="Hai tháng liên tiếp phải giả định. Sai số tích luỹ và kết quả có thể đảo "
                 "chiều hoàn toàn. Chỉ nên xem như kịch bản tham khảo, KHÔNG dùng để "
                 "điều tiết hồ chứa hay lên kế hoạch mùa vụ."),
}


# ==========================================
# TẢI MÔ HÌNH / DỮ LIỆU
# ==========================================
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(MODEL_DIR / "xgboost_drought_model.json")
    features = pd.read_csv(MODEL_DIR / "features_list.csv", header=None)[0].tolist()
    return model, features


@st.cache_data
def build_latest_rows():
    """
    Dựng dòng đặc trưng cho tháng MỚI NHẤT (tháng chưa có nhãn).

    build_dataset.py gọi dropna() nên đã xoá dòng này; thiếu nó thì app chỉ dự báo
    được tới tháng cuối cùng của dataset thay vì tháng kế tiếp thực tế.
    """
    if not INTERIM_NC.exists() or not INDICES_NC.exists():
        return pd.DataFrame()

    features = pd.read_csv(MODEL_DIR / "features_list.csv", header=None)[0].tolist()

    ds = xr.merge([xr.open_dataset(INTERIM_NC), xr.open_dataset(INDICES_NC)])
    df = ds.to_dataframe().reset_index().sort_values(["lat", "lon", "time"]).reset_index(drop=True)
    ds.close()

    # groupby().shift() giữ nguyên mọi cột (kể cả lat/lon) và không gây cảnh báo pandas
    g = df.groupby(["lat", "lon"], sort=False)
    df["target"] = g["SPEI_3"].shift(-1)
    for lag in (1, 2, 3):
        df[f"tp_lag{lag}"] = g["tp"].shift(lag)
        df[f"t2m_lag{lag}"] = g["t2m"].shift(lag)
        df[f"spei3_lag{lag}"] = g["SPEI_3"].shift(lag)

    latest = df[df["target"].isna() & df[features].notna().all(axis=1)].copy()
    latest["is_drought"] = 0
    return latest


@st.cache_data
def load_data():
    frames = [pd.read_csv(DATA_DIR / f"{s}_dataset.csv") for s in ("train", "val", "test")]

    extra = DATA_DIR / "latest_dataset.csv"
    if extra.exists():
        frames.append(pd.read_csv(extra))

    latest = build_latest_rows()
    if len(latest):
        frames.append(latest)

    full_df = pd.concat(frames, ignore_index=True)
    full_df["time"] = pd.to_datetime(full_df["time"])
    return (full_df
            .sort_values(by=["time", "lat", "lon"])
            .drop_duplicates(subset=["time", "lat", "lon"], keep="last")
            .reset_index(drop=True))


@st.cache_data
def load_climatology():
    """Khí hậu trung bình theo (ô lưới, tháng) trên giai đoạn chuẩn 1984–2013."""
    cdf = pd.read_csv(DATA_DIR / "train_dataset.csv")
    cdf["time"] = pd.to_datetime(cdf["time"])
    cdf = cdf[cdf["time"].dt.year <= 2013]
    cdf["month"] = cdf["time"].dt.month
    return (cdf.groupby(["lat", "lon", "month"])[BASE_COLS]
            .mean().reset_index())


def climb(rows, month, clim):
    """Trả về giá trị khí hậu trung bình của tháng `month` tại từng ô, đúng thứ tự rows."""
    c = clim[clim["month"] == month].drop(columns="month")
    return rows[["lat", "lon"]].merge(c, on=["lat", "lon"], how="left")


def scatter_map(dfp, title):
    fig = px.scatter(
        dfp, x="lon", y="lat", color="Xác suất hạn",
        color_continuous_scale="YlOrRd", range_color=(0, 1),
        labels={"lon": "Kinh độ", "lat": "Vĩ độ", "Xác suất hạn": "Xác suất"},
    )
    fig.update_traces(marker=dict(size=9, line=dict(width=0.4, color="black")))
    fig.update_layout(
        height=620, title=title,
        xaxis_title="Kinh độ (Lon)", yaxis_title="Vĩ độ (Lat)",
        coloraxis_colorbar=dict(title="Xác suất hạn (0-1)"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🌵 Hệ thống Dự báo Hạn hán khu vực Nam Trung Bộ")
st.markdown("Ứng dụng Trí tuệ Nhân tạo (XGBoost) dựa trên dữ liệu khí tượng ERA5 và chỉ số SPEI.")

st.sidebar.title("📌 Danh mục")
menu = st.sidebar.radio("Chọn chức năng:", ["📊 Xem Dự báo", "⚙️ Cập nhật Dữ liệu & Train lại"])

try:
    model, features = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Lỗi tải dữ liệu/mô hình: {e}. Hãy chắc chắn bạn đã chạy run_pipeline.py ít nhất 1 lần.")
    st.stop()

# ==========================================
# CHỨC NĂNG 1: XEM DỰ BÁO
# ==========================================
if menu == "📊 Xem Dự báo":
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Chọn thời gian dự báo")

    latest_date = pd.Timestamp(sorted(df["time"].unique())[-1])
    st.sidebar.caption(f"Dữ liệu mới nhất: **{latest_date:%m/%Y}**")

    default_target = latest_date + pd.DateOffset(months=1)
    st.sidebar.write("Chọn tháng cần dự báo (t+1):")
    c_a, c_b = st.sidebar.columns(2)
    ty = c_a.number_input("Năm", min_value=1984, max_value=int(default_target.year) + 1,
                          value=int(default_target.year), step=1)
    tm = c_b.number_input("Tháng", min_value=1, max_value=12,
                          value=int(default_target.month), step=1)

    target_date = pd.Timestamp(year=int(ty), month=int(tm), day=1)
    horizon = st.sidebar.slider(
        "Số tháng dự báo:", 1, 3, 1,
        help="t+1 dùng số liệu thật; t+2, t+3 dùng khí hậu trung bình cho tháng chưa biết, "
             "độ tin cậy giảm dần.",
    )

    input_date = target_date - pd.DateOffset(months=1)
    st.info(
        f"🔍 **Dữ liệu đầu vào: tháng {input_date:%Y-%m}** → dự báo cho "
        f"**{target_date:%Y-%m}** (t+1)."
    )

    rows = df[df["time"] == input_date].copy()

    if len(rows) == 0:
        st.warning(
            f"⚠️ Chưa có dữ liệu của tháng **{input_date:%Y-%m}**. "
            f"Dữ liệu hiện có tới **{latest_date:%Y-%m}**.\n\n"
            "👉 Vào tab **'⚙️ Cập nhật Dữ liệu & Train lại'** để cập nhật."
        )
        st.stop()

    clim = load_climatology()

    # Lịch sử giá trị theo từng ô, mốc 0 = tháng đầu vào
    hist = {
        "tp":    [rows["tp"].values, rows["tp_lag1"].values,
                  rows["tp_lag2"].values, rows["tp_lag3"].values],
        "t2m":   [rows["t2m"].values, rows["t2m_lag1"].values,
                  rows["t2m_lag2"].values, rows["t2m_lag3"].values],
        "spei3": [rows["SPEI_3"].values, rows["spei3_lag1"].values,
                  rows["spei3_lag2"].values, rows["spei3_lag3"].values],
    }

    n_cells = len(rows)

    for h in range(1, horizon + 1):
        target_h = input_date + pd.DateOffset(months=h)
        conf = CONFIDENCE[h]

        if h == 1:
            X = rows[features]
        else:
            # Tháng của đặc trưng: t+h-1 (chưa biết) -> dùng khí hậu trung bình
            feat_month = input_date + pd.DateOffset(months=h - 1)
            c = climb(rows, feat_month.month, clim)

            hist["tp"] = [c["tp"].values] + hist["tp"][:3]
            hist["t2m"] = [c["t2m"].values] + hist["t2m"][:3]
            hist["spei3"] = [c["SPEI_3"].values] + hist["spei3"][:3]

            X = pd.DataFrame(index=rows.index)
            X["tp"], X["t2m"] = hist["tp"][0], hist["t2m"][0]
            for col in INDEX_COLS:
                X[col] = c[col].values
            for i in (1, 2, 3):
                X[f"tp_lag{i}"] = hist["tp"][i]
                X[f"t2m_lag{i}"] = hist["t2m"][i]
                X[f"spei3_lag{i}"] = hist["spei3"][i]
            X = X[features]

        proba = model.predict_proba(X)[:, 1]

        st.markdown("---")
        st.subheader(f"🗺️ Dự báo tháng {target_h:%Y-%m}  (t+{h})")

        msg = f"{conf['icon']} **{conf['label']}** — {conf['note']}"
        getattr(st, conf["box"])(msg)

        c1, c2, c3 = st.columns(3)
        c1.metric("Nguy cơ hạn Cao nhất", f"{proba.max()*100:.1f}%")
        c2.metric("Nguy cơ hạn Trung bình", f"{proba.mean()*100:.1f}%")
        c3.metric("Số ô cảnh báo (>50%)", f"{(proba > 0.5).sum()} / {n_cells}")

        plot_df = rows[["lat", "lon"]].assign(**{"Xác suất hạn": proba})
        st.plotly_chart(scatter_map(plot_df, f"Bản đồ hạn hán — {target_h:%Y-%m}"),
                        width="stretch")

        with st.expander(f"📊 Xem dữ liệu chi tiết tháng {target_h:%Y-%m}"):
            st.dataframe(
                plot_df.sort_values("Xác suất hạn", ascending=False).head(59),
                width="stretch",
            )

        if h == 1:
            st.caption("Bước t+1 dùng số liệu quan trắc thật nên là dự báo đáng tin nhất.")

    if horizon >= 2:
        st.markdown("---")
        with st.expander("❓ Vì sao t+2, t+3 kém tin cậy hơn t+1?"):
            st.markdown(
                """
Mô hình được huấn luyện để dự báo **một tháng tới** từ 19 đặc trưng của tháng hiện tại.

| Bước | Đặc trưng trễ (`*_lag1..3`) | Biến nền tháng đích (`tp`, `t2m`, `SPI*`, `SPEI*`) |
|---|---|---|
| t+1 | Quan trắc thật | Quan trắc thật |
| t+2 | **Chính xác** — ví dụ `tp_lag1` của tháng t+1 đúng bằng `tp` tháng t đã đo | Khí hậu trung bình 1984–2013 |
| t+3 | Một phần phải giả định | Khí hậu trung bình 1984–2013 |

Ở bước t+2, phần lớn thông tin vẫn là số liệu thật: chỉ lượng mưa và nhiệt độ của chính
tháng đích là chưa biết, nên thay bằng giá trị trung bình nhiều năm của đúng tháng đó.
Sang t+3, thêm một tháng nữa phải giả định.

**Giả định quan trọng:** mưa có thể đảo ngược trạng thái hạn chỉ trong một tháng. Vì vậy
t+2 và t+3 trả lời câu hỏi *"nếu tháng tới có điều kiện khí hậu bình thường thì nguy cơ
hạn là bao nhiêu"*, chứ không phải dự báo mưa cụ thể.
                """
            )

# ==========================================
# CHỨC NĂNG 2: CHẠY PIPELINE (CÓ TIẾN TRÌNH)
# ==========================================
elif menu == "⚙️ Cập nhật Dữ liệu & Train lại":
    st.header("Cập nhật Dữ liệu & Huấn luyện lại Mô hình")
    st.markdown("""
    Hệ thống sẽ chạy ngầm file `run_pipeline.py` bao gồm 6 bước:
    1. Tải dữ liệu ERA5
    2. Tiền xử lý
    3. Tính SPI/SPEI
    4. Tạo Dataset
    5. Train mô hình XGBoost
    6. Đánh giá
    """)

    st.warning("⏳ **Cảnh báo:** Quá trình này có thể mất từ **5 - 30 phút** do phải chờ máy chủ Copernicus. Vui lòng không tắt trình duyệt.")

    if st.button("🚀 Bắt đầu chạy Pipeline", type="primary"):
        st.session_state["is_running"] = True

    if st.session_state.get("is_running", False):
        status_container = st.empty()
        progress_bar = st.progress(0.0)
        time_container = st.empty()
        log_container = st.empty()
        logs = ""

        process = subprocess.Popen(
            [sys.executable, "run_pipeline.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )

        q = queue.Queue()

        def read_output(proc, q):
            for line in proc.stdout:
                q.put(line)
            q.put(None)

        t = threading.Thread(target=read_output, args=(process, q))
        t.daemon = True
        t.start()

        total_steps = 6
        completed_steps = 0
        start_time = time.time()

        while True:
            try:
                line = q.get(timeout=1.0)
                if line is None:
                    break

                logs += line
                log_container.code(logs[-3000:], language="bash")

                if "BẮT ĐẦU:" in line:
                    step_name = line.split("BẮT ĐẦU:")[1].split("=")[0].strip()
                    status_container.info(f"⏳ Đang chạy: **{step_name}** (Bước {completed_steps + 1}/{total_steps})")
                elif "HOÀN THÀNH:" in line:
                    completed_steps += 1
                    progress_bar.progress(completed_steps / total_steps)

            except queue.Empty:
                pass

            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            time_container.markdown(f"⏱️ **Thời gian đã chạy:** `{mins} phút {secs} giây`")

        process.wait()

        if process.returncode == 0:
            progress_bar.progress(1.0)
            status_container.success("✅ Chạy Pipeline thành công! Dữ liệu và Mô hình đã được cập nhật.")
            st.session_state["is_running"] = False

            st.cache_data.clear()
            st.cache_resource.clear()

            st.info("🔄 Đang tải lại dữ liệu mới...")
            time.sleep(2)
            st.rerun()
        else:
            status_container.error("❌ Có lỗi xảy ra trong quá trình chạy Pipeline. Vui lòng kiểm tra log bên dưới.")
            st.session_state["is_running"] = False
