"""
app.py
======
Giao diện Web App dự báo hạn hán và quản lý Pipeline (Có thanh tiến trình).
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
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
# CHỨC NĂNG 1: XEM DỰ BÁO (HỖ TRỢ ĐỆ QUY)
# ==========================================
if menu == "📊 Xem Dự báo":
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Chọn thời gian dự báo")
    
    # Lấy giới hạn thời gian
    min_year = int(df['time'].dt.year.min())
    max_year = int(df['time'].dt.year.max())

    # Người dùng nhập Tháng/Năm bắt đầu (Lấy dữ liệu gốc)
    input_year = st.sidebar.number_input("Lấy dữ liệu thực tế từ Năm:", min_value=min_year, max_value=max_year, value=max_year, step=1)
    input_month_num = st.sidebar.selectbox("Tháng:", list(range(1, 13)), index=11) # Mặc định tháng 12

    input_date = pd.to_datetime(f"{input_year}-{input_month_num:02d}-01")
    input_str = input_date.strftime('%Y-%m')

    # Người dùng chọn số tháng muốn dự báo tới
    horizon = st.sidebar.slider("Số tháng muốn dự báo (Forecast Horizon):", min_value=1, max_value=3, value=1)
    
    st.info(f"🔍 **Sử dụng dữ liệu thực tế tháng {input_str}** để dự báo cho **{horizon} tháng tiếp theo**.")

    # Lọc dữ liệu gốc
    df_current = df[df['time'] == input_date].copy()

    if len(df_current) == 0:
        st.warning(f"⚠️ Hệ thống chưa có dữ liệu của tháng **{input_str}**.")
        st.stop()

    # ---- VÒNG LẶP ĐỆ QUY ----
    forecast_results = []
    current_input = df_current.copy()

    for i in range(horizon):
        target_date = input_date + pd.DateOffset(months=i + 1)
        target_str  = target_date.strftime('%Y-%m')

        # 1. Dự đoán xác suất hạn từ dữ liệu hiện tại
        X_input = current_input[features]
        probs   = model.predict_proba(X_input)[:, 1]

        # 2. Xác định độ tin cậy dựa vào bước dự báo
        if i == 0:
            confidence      = "high"
            confidence_label = f"🟢 Độ tin cậy CAO — dựa trên dữ liệu THỰC TẾ tháng {input_str}"
            border_color    = "#28a745"   # xanh lá
        elif i == 1:
            confidence      = "medium"
            confidence_label = f"🟡 Độ tin cậy TRUNG BÌNH — dùng kết quả dự đoán T+1 làm mồi, sai số bắt đầu tích lũy"
            border_color    = "#ffc107"   # vàng
        else:
            confidence      = "low"
            confidence_label = f"🔴 Độ tin cậy THẤP — dự đoán xa (T+{i+1}), sai số lớn, chỉ mang tính tham khảo!"
            border_color    = "#dc3545"   # đỏ

        # 3. Lưu kết quả
        result_df = current_input[['lat', 'lon']].copy()
        result_df['time']            = target_date
        result_df['Xác suất hạn']   = probs
        result_df['Target_Tháng']   = target_str
        result_df['confidence']      = confidence
        result_df['confidence_label'] = confidence_label
        result_df['border_color']    = border_color
        forecast_results.append(result_df)

        # 4. Cập nhật "mồi" cho bước tiếp theo (Persistence + SPEI proxy)
        if i < horizon - 1:
            new_input = current_input.copy()
            new_input['time'] = target_date

            # Dịch chuyển các lag feature
            new_input['spei3_lag3'] = new_input['spei3_lag2']
            new_input['spei3_lag2'] = new_input['spei3_lag1']
            # Proxy SPEI: hạn → -1.0, bình thường → +0.5
            new_input['spei3_lag1'] = np.where(probs > 0.5, -1.0, 0.5)

            new_input['tp_lag3']  = new_input['tp_lag2']
            new_input['tp_lag2']  = new_input['tp_lag1']
            new_input['tp_lag1']  = new_input['tp']   # Persistence

            new_input['t2m_lag3'] = new_input['t2m_lag2']
            new_input['t2m_lag2'] = new_input['t2m_lag1']
            new_input['t2m_lag1'] = new_input['t2m']  # Persistence

            current_input = new_input

    # ---- HIỂN THỊ KẾT QUẢ ----
    df_forecast = pd.concat(forecast_results, ignore_index=True)

    st.markdown("---")
    for i, target_str in enumerate(df_forecast['Target_Tháng'].unique()):
        df_month     = df_forecast[df_forecast['Target_Tháng'] == target_str]
        max_prob     = df_month['Xác suất hạn'].max() * 100
        avg_prob     = df_month['Xác suất hạn'].mean() * 100
        conf_label   = df_month['confidence_label'].iloc[0]
        border_color = df_month['border_color'].iloc[0]

        # Tiêu đề tháng
        st.subheader(f"🗺️ Dự báo Tháng {target_str}  (T+{i+1})")

        # Banner độ tin cậy màu tương ứng
        if i == 0:
            st.success(conf_label)
        elif i == 1:
            st.warning(conf_label)
        else:
            st.error(conf_label)

        col1, col2 = st.columns(2)
        col1.metric("Nguy cơ hạn Cao nhất",   f"{max_prob:.1f}%")
        col2.metric("Nguy cơ hạn Trung bình",  f"{avg_prob:.1f}%")

        # Vẽ bản đồ — viền ngoài đổi màu theo độ tin cậy
        fig = px.scatter(
            df_month,
            x="lon", y="lat",
            color="Xác suất hạn",
            color_continuous_scale="YlOrRd",
            range_color=(0, 1),
            labels={"lon": "Kinh độ", "lat": "Vĩ độ", "Xác suất hạn": "Xác suất"},
            size=[10] * len(df_month),
            title=f"Bản đồ hạn hán — {target_str}"
        )
        fig.update_layout(
            height=500,
            xaxis_title="Kinh độ (Lon)",
            yaxis_title="Vĩ độ (Lat)",
            coloraxis_colorbar=dict(title="Xác suất hạn (0–1)"),
            # Khung viền bản đồ đổi màu theo độ tin cậy
            paper_bgcolor="white",
            plot_bgcolor="#f9f9f9",
            margin=dict(l=10, r=10, t=40, b=10),
            shapes=[dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color=border_color, width=4)
            )]
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
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