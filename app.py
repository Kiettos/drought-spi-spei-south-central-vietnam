"""
app.py
======
Giao diện Web App dự báo hạn hán và quản lý Pipeline (Có thanh tiến trình).
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
from pathlib import Path
import subprocess
import sys
import time
import threading
import queue

# Cấu hình trang web
st.set_page_config(
    page_title="Hệ thống Dự báo Hạn hán", 
    layout="wide", 
    page_icon="🌵", 
    initial_sidebar_state="expanded"
)


# ==========================================
# CÁC HÀM TẢI DỮ LIỆU & MÔ HÌNH
# ==========================================
@st.cache_resource
def load_model():
    model_dir = Path("outputs/models")
    model = xgb.XGBClassifier()
    model.load_model(model_dir / "xgboost_drought_model.json")
    features = pd.read_csv(model_dir / "features_list.csv", header=None)[0].tolist()
    return model, features

@st.cache_data
def load_data():
    data_dir = Path("data/processed")
    train_df = pd.read_csv(data_dir / "train_dataset.csv")
    val_df = pd.read_csv(data_dir / "val_dataset.csv")
    test_df = pd.read_csv(data_dir / "test_dataset.csv")
    
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df['time'] = pd.to_datetime(full_df['time'])
    full_df = full_df.sort_values(by=['time', 'lat', 'lon']).reset_index(drop=True)
    return full_df

# ==========================================
# GIAO DIỆN CHÍNH
# ==========================================
st.title("🌵 Hệ thống Dự báo Hạn hán khu vực Nam Trung Bộ")
st.markdown("Ứng dụng Trí tuệ Nhân tạo (XGBoost) dựa trên dữ liệu khí tượng ERA5 và chỉ số SPEI.")

st.sidebar.title("📌 Danh mục")
menu = st.sidebar.radio("Chọn chức năng:", ["📊 Xem Dự báo", "⚙️ Cập nhật Dữ liệu & Train lại"])

# Tải dữ liệu và mô hình
try:
    model, features = load_model()
    df = load_data()
except Exception as e:
    st.error(f"Lỗi tải dữ liệu/mô hình: {e}. Hãy chắc chắn bạn đã chạy file run_pipeline.py ít nhất 1 lần.")
    st.stop()

# ==========================================
# CHỨC NĂNG 1: XEM DỰ BÁO
# ==========================================
if menu == "📊 Xem Dự báo":
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Chọn thời gian dự báo")
    st.sidebar.write("Dùng dữ liệu tháng trước để dự báo tháng bạn chọn:")

    min_year = int(df['time'].dt.year.min())
    max_year = int(df['time'].dt.year.max()) + 1 

    selected_year = st.sidebar.number_input("Năm dự báo:", min_value=min_year, max_value=max_year, value=max_year-1, step=1)
    selected_month_num = st.sidebar.selectbox("Tháng dự báo:", list(range(1, 13)), index=0)

    target_date = pd.to_datetime(f"{selected_year}-{selected_month_num:02d}-01")
    input_date = target_date - pd.DateOffset(months=1)
    
    target_str = target_date.strftime('%Y-%m')
    input_str = input_date.strftime('%Y-%m')

    st.info(f"🔍 **Lấy dữ liệu tháng {input_str}** để dự báo cho **tháng {target_str}**.")

    df_month = df[df['time'] == input_date].copy()

    if len(df_month) == 0:
        st.warning(f"⚠️ Hệ thống chưa có dữ liệu của tháng **{input_str}**. Không thể dự báo cho tháng {target_str}.\n\n👉 Vui lòng chuyển sang tab **'⚙️ Cập nhật Dữ liệu & Train lại'** để cập nhật.")
        st.stop()

    X_month = df_month[features]
    df_month['Xác suất hạn'] = model.predict_proba(X_month)[:, 1]

    max_prob = df_month['Xác suất hạn'].max() * 100
    avg_prob = df_month['Xác suất hạn'].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Tháng dự báo", target_str)
    col2.metric("Nguy cơ hạn Cao nhất", f"{max_prob:.1f}%")
    col3.metric("Nguy cơ hạn Trung bình", f"{avg_prob:.1f}%")

    st.markdown("---")

    st.subheader(f"🗺️ Bản đồ Dự báo Hạn hán cho Tháng {target_str}")
    fig = px.scatter(
        df_month, 
        x="lon", y="lat", color="Xác suất hạn",
        color_continuous_scale="YlOrRd", range_color=(0, 1),
        labels={"lon": "Kinh độ", "lat": "Vĩ độ", "Xác suất hạn": "Xác suất"},
    )
    # Kích thước điểm cố định + khoá tỉ lệ trục để bản đồ không bị bóp méo.
    # Thiếu scaleanchor thì 59 điểm bị dồn thành các cột dọc, mất hình chữ S.
    fig.update_traces(marker=dict(size=9, line=dict(width=0.4, color="black")))
    fig.update_layout(
        height=700,
        xaxis_title="Kinh độ (Lon)", yaxis_title="Vĩ độ (Lat)",
        coloraxis_colorbar=dict(title="Xác suất hạn (0-1)"),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("📊 Xem dữ liệu chi tiết (Dạng bảng)"):
        st.dataframe(
            df_month[['time', 'lat', 'lon', 'Xác suất hạn', 'target']]
            .sort_values(by='Xác suất hạn', ascending=False).head(50),
            width='stretch'
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
        st.session_state['is_running'] = True

    if st.session_state.get('is_running', False):
        # Khởi tạo các vùng hiển thị
        status_container = st.empty()
        progress_bar = st.progress(0.0)
        time_container = st.empty()
        
        log_container = st.empty()
        logs = ""
        
        # Khởi tạo tiến trình chạy run_pipeline.py
        process = subprocess.Popen(
            [sys.executable, "run_pipeline.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        
        # Hàm đọc output từ process (chạy trong 1 luồng riêng để không block UI)
        q = queue.Queue()
        def read_output(proc, q):
            for line in proc.stdout:
                q.put(line)
            q.put(None) # Kết thúc
            
        t = threading.Thread(target=read_output, args=(process, q))
        t.daemon = True
        t.start()
        
        total_steps = 6
        completed_steps = 0
        start_time = time.time()
        
        while True:
            try:
                # Đọc từng dòng log với timeout 1 giây
                line = q.get(timeout=1.0)
                if line is None:
                    break # Process kết thúc
                    
                logs += line
                log_container.code(logs[-3000:], language="bash") # Hiển thị 3000 ký tự cuối
                
                # Cập nhật thanh tiến trình dựa vào log
                if "BẮT ĐẦU:" in line:
                    step_name = line.split("BẮT ĐẦU:")[1].split("=")[0].strip()
                    status_container.info(f"⏳ Đang chạy: **{step_name}** (Bước {completed_steps + 1}/{total_steps})")
                elif "HOÀN THÀNH:" in line:
                    completed_steps += 1
                    progress_bar.progress(completed_steps / total_steps)
                    
            except queue.Empty:
                pass # Nếu không có log mới, tiếp tục vòng lặp để cập nhật thời gian
                
            # Cập nhật đồng hồ đếm thời gian
            elapsed = int(time.time() - start_time)
            mins, secs = divmod(elapsed, 60)
            time_container.markdown(f"⏱️ **Thời gian đã chạy:** `{mins} phút {secs} giây`")
            
        process.wait()
        
        if process.returncode == 0:
            progress_bar.progress(1.0)
            status_container.success("✅ Chạy Pipeline thành công! Dữ liệu và Mô hình đã được cập nhật.")
            st.session_state['is_running'] = False
            
            st.cache_data.clear()
            st.cache_resource.clear()
            
            st.info("🔄 Đang tải lại dữ liệu mới...")
            time.sleep(2)
            st.rerun()
        else:
            status_container.error("❌ Có lỗi xảy ra trong quá trình chạy Pipeline. Vui lòng kiểm tra log bên dưới.")
            st.session_state['is_running'] = False