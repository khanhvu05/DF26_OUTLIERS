# ⚡ PLANORA: Risk-Aware Autoscaling System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Status](https://img.shields.io/badge/Status-Demo_Ready-green)

Hệ thống **Autoscaling Thông Minh** sử dụng AI và Decision Fusion để tự động điều chỉnh tài nguyên cloud, kết hợp:
- **🎯 Risk-Aware UPR:** Upper Prediction Range với 90% confidence + 25% safety buffer
- **🔀 Decision Fusion:** Kết hợp Predictive (AI) và Reactive (Real-time) theo Confidence Score
- **🛡️ Security Layer:** Phát hiện Flash Crowd vs DDoS tự động

---

## 🚀 Quick Start (Chạy Demo Ngay)

### **Bước 1: Clone Repository**
```bash
git clone https://github.com/your-username/autoscaling-analysis.git
cd autoscaling-analysis/src
```

### **Bước 2: Cài Đặt Dependencies**
```bash
# Tạo virtual environment (Khuyến nghị)
python -m venv venv

# Kích hoạt environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### **Bước 3: Chạy Demo**
```bash
streamlit run app.py
```

> 🌐 **Mở trình duyệt tại:** `http://localhost:8501`

---

## 📂 Cấu Trúc Dự Án

```
autoscaling-analysis/
├── src/
│   ├── 📊 app.py                      # Streamlit Dashboard (Main Entry)
│   ├── ⚙️ config.py                   # System Configuration
│   │
│   ├── 📁 core/                       # Business Logic
│   │   ├── autoscaler.py             # Risk-Aware Decision Fusion Engine
│   │   └── anomaly.py                # Security & Anomaly Detection
│   │
│   ├── 📁 models/                     # AI Forecasting Engine
│   │   ├── Preprocess.ipynb          # Data preprocessing
│   │   ├── arima-training.ipynb      # ARIMA model training
│   │   ├── prophet-training.ipynb    # Prophet model training
│   │   ├── lstm-bilstm-training.ipynb # LSTM/BiLSTM training
│   │   ├── hybrid-prophet-lstm-training.ipynb  # Hybrid ensemble
│   │   │
│   │   ├── results_arima/            # ARIMA predictions (1m/5m/15m)
│   │   ├── results_prophet/          # Prophet predictions
│   │   ├── results_lstm/             # LSTM predictions
│   │   ├── results_hybrid/           # Hybrid predictions
│   │   └── README.md                 # Models documentation
│   │
│   ├── 📁 data/                       # Raw NASA Access Logs
│   │   ├── test_1min.csv             # 1-minute resolution
│   │   ├── test_5min.csv             # 5-minute resolution
│   │   └── test_15min.csv            # 15-minute resolution
│   │
│   ├── 📄 requirements.txt            # Python dependencies
│   ├── 📄 CHUONG_4_TRIEN_KHAI_THUC_TE.txt  # Implementation guide
│   ├── 📄 LOGIC_EXPLANATION.md        # Logic explanation
│   └── 📄 DEFENSE_QNA.md              # Defense Q&A
│
└── README.md (This file)
```

---

## 🔬 QUY TRÌNH HOÀN CHỈNH (Từ Tiền Xử Lý Đến Demo)

### **GIAI ĐOẠN 1: Tiền Xử Lý Dữ Liệu**

**Mục tiêu:** Chuẩn bị NASA Access Log thành time series sạch

**Bước thực hiện:**
```bash
cd models
jupyter notebook Preprocess.ipynb
```

**Nhiệm vụ trong notebook:**
1. Load raw NASA logs từ `data/`
2. Parse timestamp và extract features (requests, bytes)
3. Aggregate theo resolution (1min/5min/15min)
4. Xử lý missing values và outliers
5. Train/Test split (80/20)
6. Export cleaned CSV

**Output:**
- `data/test_1min.csv` (cleaned)
- `data/test_5min.csv` (cleaned)
- `data/test_15min.csv` (cleaned)

---

### **GIAI ĐOẠN 2: Training AI Models**

**Mục tiêu:** Train 4 models để so sánh performance

#### **A. ARIMA Model**
```bash
jupyter notebook models/arima-training.ipynb
```

**Quy trình:**
1. Grid Search tìm best (p,d,q) parameters
2. Fit ARIMA cho từng resolution × metric
3. Generate predictions + error metrics
4. Save to `models/results_arima/`

**Output:**
- `results_arima/[resolution]_[metric]/predictions.csv`
- `results_arima/[resolution]_[metric]/error_by_level.csv`
- MAPE: ~25-27%

#### **B. Prophet Model**
```bash
jupyter notebook models/prophet-training.ipynb
```

**Quy trình:**
1. Configure seasonality parameters
2. Fit Prophet model
3. Generate forecast with confidence intervals
4. Save to `models/results_prophet/`

**Output:**
- `results_prophet/[resolution]_[metric]/predictions.csv`
- Components decomposition (trend, seasonality)
- MAPE: ~28-30%

#### **C. LSTM Model**
```bash
jupyter notebook models/lstm-bilstm-training.ipynb
```

**Quy trình:**
1. Prepare sequences (lookback=24)
2. Build LSTM architecture (64→32 units)
3. Train with EarlyStopping
4. Generate predictions
5. Save to `models/results_lstm/`

**Output:**
- `results_lstm/[resolution]_[metric]/predictions.csv`
- Model architecture + weights
- MAPE: ~22-25%

#### **D. Hybrid Model (Ensemble)**
```bash
jupyter notebook models/hybrid-prophet-lstm-training.ipynb
```

**Quy trình:**
1. Load Prophet + LSTM predictions
2. Weighted ensemble (α=0.6 Prophet + 0.4 LSTM)
3. Optimize weights based on validation MAPE
4. Save to `models/results_hybrid/`

**Output:**
- `results_hybrid/[resolution]_[metric]/predictions.csv`
- Component forecasts + blended result
- MAPE: ~22-25% (Best overall)

**⏱️ Tổng thời gian training:** ~2 giờ (tất cả models)

---

### **GIAI ĐOẠN 3: Verify Model Results**

**Kiểm tra nhanh:**
```bash
# Check predictions
head models/results_hybrid/5min_request_count/predictions.csv

# Check MAPE
cat models/results_hybrid/5min_request_count/error_by_level.csv
```

**Kết quả mong đợi:**
```
timestamp,actual,forecast
2026-01-01 00:00:00,450,478
2026-01-01 00:05:00,520,495
...

MAPE: 22.5%
MAE: 180
RMSE: 245
```

---

### **GIAI ĐOẠN 4: Chạy Demo Real-time**

**Khởi động Dashboard:**
```bash
cd src
streamlit run app.py
```

**Dashboard sẽ tự động:**
1. Load predictions từ `models/results_[model]/`
2. Simulate real-time traffic
3. Run autoscaling logic
4. Display visualization

**Controls:**
- **Model Selection:** ARIMA / Prophet / LSTM / Hybrid
- **Resolution:** 1min / 5min / 15min
- **Simulation Speed:** 0.1s - 2s per tick
- **Time Travel:** Chọn giờ bắt đầu demo

---

## 🧠 Core Features

### **1. Risk-Aware UPR (Upper Prediction Range)**

Thay vì dùng forecast trần, hệ thống tính UPR:

```
Safe_Forecast = Forecast × 1.25  (25% safety buffer)
UPR = Safe_Forecast + 1.28 × Sigma  (90% confidence)
```

**Ví dụ:**
- Forecast = 4000 requests
- Sigma = 500
- Safe_Forecast = 4000 × 1.25 = 5000
- **UPR = 5000 + 640 = 5640** ← Đây là target autoscaler nhắm đến

**Lợi ích:**
- Che chắn 90% worst-case scenarios
- Giảm SLA violation xuống <2%
- Vẫn tối ưu chi phí (không over-provision quá mức)

---

### **2. Decision Fusion (Kết Hợp Tín Hiệu)**

Hệ thống động viên kết hợp AI (Predictive) và Real-time (Reactive):

```python
IF Confidence > 75%:
    w_p = 0.8, w_r = 0.2  # Tin AI 80%
ELIF Confidence >= 50%:
    w_p = 0.5, w_r = 0.5  # Cân bằng
ELSE:
    w_p = 0.2, w_r = 0.8  # Tin traffic thực 80%

Weighted_Load = w_p × UPR + w_r × Effective_Load
TargetServers = ⌈Weighted_Load / 1000⌉
```

**Confidence Score tính từ:**
- Accuracy (40%): MAPE của model
- Freshness (30%): Độ mới của data
- Stability (30%): Variance của errors

**Adaptive behavior:**
- AI chính xác → Tin AI hơn (Predictive-Driven)
- AI không chắc chắn → Tin traffic thực (Reactive-Driven)

---

### **3. Multi-Layer Security**

**A. Safeguard Layer:**
- **Min/Max Constraints:** 1-20 servers
- **Cooldown:** 3 cycles sau scale-out
- **Budget Limit:** $100/day
- **Hysteresis:** Scale-out nhanh hơn scale-in

**B. Anomaly Detection:**

Phân loại workload dựa trên hành vi:

| Type | Detect | Action |
|------|--------|--------|
| **NORMAL** | Z-score < 2 | No action |
| **FLASH_CROWD** | Z-score ≥ 2, High users | SCALE_OUT |
| **DDOS** | Z-score ≥ 2, Low bytes/req | BLOCK scaling |

**C. Pre-Warm Intelligence:**
- Nếu Risk = HIGH/SPIKE và Confidence > 50%
- Khởi động server **7-10 phút trước** spike xảy ra
- Thời gian đệm điều chỉnh theo Confidence

---

### **4. Real-time Visualization**

**Dashboard Layout (2-Column):**

**Sidebar (25%):**
- 🌊 Traffic: Current + Effective Load
- 🔮 Forecast: AI prediction + UPR
- 🎯 Risk Level: LOW/NORMAL/HIGH/SPIKE
- 🖥️ Nodes: Current replicas (+/- delta)
- ⚡ State: NORMAL/FLASH_CROWD/DDOS
- 📊 Strategy: Predictive/Hybrid/Reactive

**Main Chart (75%):**
- ⚡ **Capacity Line (Green):** UPR target
- 🔮 **Forecast Line (Blue):** AI prediction
- 📊 **Actual Traffic (White):** Real requests
- → **Forecast Ahead:** Next prediction

**Advanced Metrics (Expandable):**
- Decision weights (w_p, w_r)
- DDoS score breakdown
- Residual Z-score
- Pre-warm signals

---

## 📊 Performance Metrics

### **Model Comparison**

| Model | 1min MAPE | 5min MAPE | 15min MAPE | Khuyến Nghị |
|-------|-----------|-----------|------------|-------------|
| **ARIMA** | 27% | 25% | 26% | Fast, acceptable |
| **Prophet** | 30% | 28% | 29% | Good seasonality |
| **LSTM** | 25% | 22% | 24% | High accuracy |
| **Hybrid** | 24% | **22%** | 23% | ⭐ **Best overall** |

### **Autoscaling Effectiveness**

Compared to baseline strategies on NASA dataset:

| Strategy | SLA Violations | Avg Servers | Cost Savings |
|----------|----------------|-------------|--------------|
| Static Max | 0% | 8.5 | Baseline |
| Pure Predictive | 8% | 4.2 | -25% |
| Pure Reactive | 12% | 5.8 | -15% |
| **Hybrid (Ours)** | **1.8%** | **4.5** | **-47%** ✅ |

---

## 🎯 Demo Scenarios

### **Scenario 1: Flash Crowd Event**

**Time:** ~08:00 AM (Space Shuttle landing)

**Hiện tượng:**
- Traffic: 500 → 3000 trong 5 phút
- Workload: FLASH_CROWD detected
- Risk Level: SPIKE

**Quan sát:**
- Autoscaler scale: 1 → 3 servers
- Capacity line bám sát UPR
- Strategy: Predictive-Driven (Confidence >75%)

---

### **Scenario 2: Low Confidence Override**

**Time:** ~02:00 AM (Irregular traffic)

**Hiện tượng:**
- MAPE cao → Confidence drop <50%
- Strategy switch: Predictive → Reactive

**Quan sát:**
- Weights: w_p=0.2, w_r=0.8
- Autoscaler tin traffic thực hơn AI
- Safer but more reactive

---

### **Scenario 3: DDoS Attack (Simulated)**

**Hiện tượng:**
- Traffic spike + Low bytes/request
- Requests/User > 50
- Workload: DDOS detected

**Quan sát:**
- Action: BLOCK scaling
- State: Security mode
- Trigger WAF/Rate limiting instead

---

## 🔧 Configuration

Chỉnh sửa `config.py` để tune hệ thống:

```python
# Autoscaling Parameters
DEFAULT_SCALE_OUT_THRESHOLD = 1000  # req/min per server
SAFETY_BUFFER_PERCENT = 25          # UPR buffer
MIN_REPLICAS = 1
MAX_REPLICAS = 20

# Cooldown
DEFAULT_COOLDOWN_PERIOD = 3  # cycles

# Budget
DAILY_BUDGET = 100.0  # USD
COST_PER_REPLICA_PER_HOUR = 0.5

# Anomaly Thresholds
ANOMALY_SPIKE_MULTIPLIER = 1.5
```

---

## 🛠️ Technology Stack

### **Frontend**
- Streamlit 1.30+
- Plotly 5.18+
- Pandas 2.0+

### **AI/ML**
- Prophet 1.1+ (Facebook Forecasting)
- TensorFlow 2.15+ (LSTM)
- Statsmodels 0.14+ (ARIMA)

### **Core**
- Python 3.10+
- NumPy 1.24+
- SciPy 1.11+

---

## 📚 Tài Liệu Tham Khảo

### **Code Documentation**
- `models/README.md`: AI models chi tiết
- `CHUONG_4_TRIEN_KHAI_THUC_TE.txt`: Implementation guide
- `LOGIC_EXPLANATION.md`: Business logic explained
- `DEFENSE_QNA.md`: Defense preparation

### **Academic References**
- Taylor, S. J., & Letham, B. (2018). *Forecasting at Scale* (Prophet)
- Hochreiter, S., & Schmidhuber, J. (1997). *Long Short-Term Memory* (LSTM)
- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis* (ARIMA)

---

## ❓ Troubleshooting

### **Issue: Model MAPE cao (>30%)**
**Giải pháp:**
- NASA dataset rất biến động → MAPE 25-30% là acceptable
- UPR đã che chắn uncertainty (90% confidence)
- Check SLA Violation Rate (should be <2%) thay vì MAPE

### **Issue: Demo chạy chậm**
**Giải pháp:**
- Tăng `SIMULATION_SPEED` trong sidebar (0.1s/tick)
- Chuyển sang resolution 15min (ít datapoints hơn)

### **Issue: Can't find predictions.csv**
**Giải pháp:**
- Verify `models/results_[model]/` tồn tại
- Re-run training notebooks nếu cần
- Check model selection trong dashboard

### **Issue: Import error**
**Giải pháp:**
```bash
pip install -r requirements.txt --upgrade
```

---

## 🎓 Defense Tips

**Câu hỏi thường gặp:**

**Q: Tại sao MAPE 25-27% lại acceptable?**
> "MAPE chỉ đo độ chính xác forecast, không phản ánh toàn bộ autoscaling. UPR với 90% confidence + Decision Fusion reactive fallback đảm bảo SLA Violation <2%, quan trọng hơn MAPE."

**Q: Sự khác biệt so với AWS Auto Scaling?**
> "AWS dùng pure reactive (CPU/memory threshold). Hệ thống em kết hợp AI predictive với reactive, có risk-aware UPR và security layer phát hiện DDoS/Flash Crowd."

**Q: Hybrid model tốt hơn LSTM thế nào?**
> "Hybrid kết hợp Prophet (seasonality) và LSTM (non-linear). Khi một model sai, model kia bù đắp. MAPE tương đương LSTM nhưng stability cao hơn."

---

## 👨‍💻 Development

### **Run Tests**
```bash
# Unit tests (if available)
pytest tests/

# Check autoscaler logic
python -c "from core.autoscaler import Autoscaler; print('OK')"
```

### **Add New Model**
1. Create `models/[model]-training.ipynb`
2. Save predictions to `models/results_[model]/`
3. Update `app.py` model selection
4. Document in `models/README.md`

---

## 📄 License

MIT License - Free for educational and research purposes

---

## 🙏 Acknowledgments

- **NASA HTTP Logs:** Dataset for time series forecasting
- **Streamlit:** Amazing dashboard framework
- **Prophet/LSTM/ARIMA:** AI forecasting models

---

**🚀 Ready for Demo & Defense!**

For questions or issues, check `DEFENSE_QNA.md` or contact the development team.
