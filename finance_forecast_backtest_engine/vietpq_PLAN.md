# 📈 Backtesting Engine - Implementation Plan (VN Market)

## 🎯 1. Mục tiêu (Goal & Scope)

Xây dựng một hệ thống Backtest mô phỏng giao dịch cổ phiếu (ví dụ: ACB) dựa trên dự báo của mô hình AI (DEFAULT_ANN hoặc LSTM). Hệ thống sẽ "quay ngược thời gian", duyệt qua từng ngày trong quá khứ, đưa ra quyết định Mua/Bán/Nắm giữ dựa trên dự đoán của AI, đồng thời tuân thủ nghiêm ngặt các quy định thực tế của thị trường chứng khoán Việt Nam (HOSE).

**Phạm vi (Scope):**

* **Input:** Tập dữ liệu lịch sử OHLCV (Open, High, Low, Close, Volume) và các mô hình AI đã được train.
* **Output:** Báo cáo hiệu suất đầu tư (Return, Drawdown, Win Rate) và chi tiết lịch sử giao dịch.
* **Giới hạn hiện tại:** Chỉ chạy trên khung thời gian Ngày (Daily Chart). Không margin (vay nợ), không bán khống (short selling).

---

## 📊 2. Cấu trúc Dữ liệu & Tham số Hệ thống (Inputs & Parameters)

Để các module giao tiếp trơn tru, dưới đây là định nghĩa về dữ liệu đầu vào và các tham số cấu hình cho toàn bộ Engine.

### 2.1. Dữ liệu Đầu vào (Raw Data Sample)

Hệ thống sử dụng file CSV (ví dụ: `VN30_Dataset_2015_2026.csv`). Dữ liệu này chỉ chứa các ngày làm việc (đã bỏ qua cuối tuần và ngày lễ).

```csv
time,open,high,low,close,volume,Ticker
2014-06-30,2.35,2.37,2.34,2.35,272267,ACB
2014-07-01,2.37,2.37,2.35,2.37,203292,ACB
2014-07-02,2.37,2.37,2.35,2.37,114809,ACB
2014-07-03,2.37,2.38,2.35,2.38,309194,ACB
2014-07-04,2.38,2.38,2.37,2.37,117744,ACB

```

### 2.2. Bảng Tham số Cấu hình (Config Parameters)

Các tham số này nên được gom vào một file `config.py` hoặc truyền qua Argument Parser khi chạy CLI.

| Tham số | Ý nghĩa / Giải thích | Ví dụ Mặc định |
| --- | --- | --- |
| `initial_capital` | Số tiền mặt ban đầu cấp cho danh mục. | `100_000_000` (VNĐ) |
| `transaction_fee` | Phí giao dịch tính trên mỗi chiều Mua và Bán. | `0.0015` (0.15%) |
| `lookback_window` | Số ngày dữ liệu quá khứ cần thiết để AI tạo input dự báo. | `30` (ngày) |
| `prediction_days` | Số ngày trong tương lai mà mô hình xuất ra dự báo. | `3` (T+1, T+2, T+3) |
| `stop_loss_threshold` | Ngưỡng cắt lỗ tự động (Bán bất chấp tín hiệu AI). | `-0.07` (-7%) |
| `lot_size` | Đơn vị lô cổ phiếu quy định của sàn giao dịch (HOSE). | `100` |

---

## 📚 3. Định nghĩa & Luật chơi thực tế (Market Rules)

Đây là các quy tắc Hard-coded bắt buộc phải tuân thủ để đảm bảo Backtest không sinh ra "lãi ảo".

### 3.1. Quy tắc Thanh toán T+2.5 & Xử lý Ngày nghỉ (Calendar Logic)

Ở Việt Nam, giao dịch không hoàn tất ngay lập tức. Hệ thống phải theo dõi "trạng thái chờ" của Cổ phiếu và Tiền.

**⚠️ Hố đen Ngày nghỉ (Calendar Logic):**

* **Tuyệt đối KHÔNG ĐƯỢC:** Dùng phép cộng ngày tháng (`current_date + 2 days`). Vì thị trường nghỉ Thứ 7, Chủ Nhật và Lễ Tết. Mua Thứ 6 thì hàng về Thứ 3 tuần sau, không phải Chủ Nhật.
* **Cách giải quyết (Array Indexing):** Sử dụng **Index** của mảng dữ liệu. Vì dữ liệu Raw đã loại bỏ ngày nghỉ, nên `Ngày T+2` đơn giản chính là `data[current_index + 2]`.

**Bảng Ví dụ Cụ thể Logic T+2.5 (Dùng Array Index):**

| Thời gian thực tế | Index Dữ liệu | Hành động Hệ thống | Trạng thái Tiền / Cổ phiếu |
| --- | --- | --- | --- |
| **Tối Thứ 5** (Ngày $T-1$) | `i - 1` | AI báo tín hiệu **MUA**. | Chưa có thay đổi. |
| **Sáng Thứ 6** (Ngày $T$) | `i` | Khớp lệnh **MUA 1000 ACB**. | Tiền khả dụng bị trừ ngay. Cổ phiếu nằm ở ví `pending_shares`. |
| **Thứ 2 tuần sau** (Ngày $T+1$) | `i + 1` | Hold. | Cổ phiếu vẫn ở `pending`. Chưa được phép bán. |
| **Thứ 3 tuần sau** (Ngày $T+2$) | `i + 2` | Chuyển trạng thái. | Cổ phiếu được tự động chuyển từ `pending` sang `available_shares` để có thể **BÁN**. |

### 3.2. Cơ chế Khớp lệnh & Tránh bẫy Lookahead Bias (Rất quan trọng)

Để mô phỏng chính xác nhất hành vi giao dịch thực tế trên sàn (phiên ATO/ATC), chúng ta sẽ mặc định dùng giá **Open** của ngày hôm sau để khớp lệnh thay vì giá Close.

* **Vấn đề (Lookahead Bias):** Nếu lấy dự báo lúc tối ngày $T$ và đặt lệnh bằng giá Close của chính ngày $T$, đó là quay ngược thời gian.
* **Giải pháp (Open Price Execution):**
1. Tối Ngày $T$: Chạy mô hình AI với dữ liệu giá Close ngày $T$. AI đưa ra tín hiệu.
2. Sáng Ngày $T+1$: Hệ thống tự động mô phỏng việc đặt lệnh ATO ngay đầu phiên.
3. **Giá khớp lệnh:** Hệ thống lấy giá `Open[T+1]` để tính toán khối lượng mua/bán thực tế.



### 3.3. Khối lượng giao dịch (Lot Size)

Sàn HOSE quy định chỉ được phép giao dịch bội số của 100 cổ phiếu (lô chẵn).

* **Công thức tính số cổ phiếu tối đa có thể mua:**

```python
# 1. Tính số lượng lý thuyết có thể mua bao gồm cả phí
raw_shares = available_cash / (price_open_T1 * (1 + transaction_fee))

# 2. Làm tròn xuống bội số của 100 (Luật lô chẵn HOSE)
actual_shares = (int(raw_shares) // 100) * 100

```

### 3.4. Chi phí giao dịch (Transaction Cost)

* **Phí quy định:** **0.15%** cho mỗi chiều Mua và chiều Bán (Đã bao gồm phí môi giới, thuế phí Sở...).
* Hệ thống sẽ trừ tiền theo công thức: `Total_Cost = actual_shares * price * 1.0015`.

---

## 🛡️ 4. Cơ chế Quản trị Rủi ro (Stop Loss Override)

Không thể phó mặc 100% cho AI trong các biến cố "Thiên nga đen" (thị trường sập do tin vĩ mô). Hệ thống cần một "Van an toàn".

* **Quy tắc Ghi đè (Override Rule):**
Cuối mỗi ngày $T$, hệ thống tính:
`Lợi_nhuận_hiện_tại = (Close[T] - Giá_vốn_trung_bình) / Giá_vốn_trung_bình`
* **Stop Loss (-7%):** Nếu `Lợi_nhuận_hiện_tại <= -0.07`, hệ thống sẽ bỏ qua mọi dự báo "Tăng" của AI, và **bắt buộc tạo tín hiệu BÁN TOÀN BỘ** vào sáng ngày $T+1$ (Khớp bằng giá `Open[T+1]`).

---

## 🎮 5. Chiến lược giao dịch (Trading Strategy 1 - Conservative)

Sử dụng kết hợp dự báo AI (`pred_t1`, `pred_t2`, `pred_t3`) và Quản trị rủi ro. Các quy tắc sẽ được xét theo thứ tự ưu tiên:

1. **ƯU TIÊN 1: CẮT LỖ (STOP LOSS)**
* Nếu đang cầm cổ phiếu (đã qua T+2) VÀ Lãi/Lỗ hiện tại $\le -7\%$.
* **Hành động:** BÁN TOÀN BỘ.


2. **ƯU TIÊN 2: TÍN HIỆU MUA (BUY)**
* Cả 3 dự báo đều cao hơn giá Đóng cửa hiện tại: `pred_t1 > Close_T` AND `pred_t2 > Close_T` AND `pred_t3 > Close_T`.
* Đang không nắm giữ cổ phiếu, có sẵn Tiền.
* **Hành động:** MUA toàn bộ bằng Tiền mặt khả dụng.


3. **ƯU TIÊN 3: TÍN HIỆU BÁN (SELL)**
* Có ít nhất 2 trong 3 dự báo thấp hơn giá Đóng cửa hiện tại (Đa số vote Giảm).
* Đang nắm giữ Cổ phiếu khả dụng.
* **Hành động:** BÁN TOÀN BỘ.


4. **ƯU TIÊN 4: ĐỨNG NGOÀI (HOLD)**
* Không thỏa mãn các điều kiện trên (tín hiệu nhiễu, hoặc đang đợi Tiền/Cổ phiếu về).



---

## 🏗️ 6. Kiến trúc Hệ thống (System Architecture)

```text
finance_forecast_backtest_engine/
├── __init__.py
├── data_loader.py       # Load CSV, xử lý features, batch_predict để tối ưu hiệu năng
├── portfolio.py         # Quản lý Cash, Shares, trạng thái T+2 (pending vs available)
├── strategy.py          # Implement logic Chiến lược 1 & Stop Loss
├── backtest.py          # Vòng lặp thời gian chính (Time-loop simulation)
└── metrics.py           # Tính toán Sharpe, Drawdown, Return, Win Rate

run_backtest.py          # File chạy chính qua CLI

```

---

## 📋 7. Các bước Thực thi Chi tiết (Implementation Steps)

### Bước 1: Xử lý Dữ liệu & Dự báo Batch (Tối ưu hiệu năng)

*File: `data_loader.py*`

* Quét qua `df`, tạo một mảng Numpy 3D chứa toàn bộ cửa sổ 30 ngày.
* Gọi `predictions_array = model.predict(all_features)` **chỉ 1 lần duy nhất** trước vòng lặp Backtest để tránh thắt cổ chai hiệu năng. Map kết quả này vào DataFrame theo từng Index tương ứng.

### Bước 2: Xây dựng Portfolio Manager (Xử lý Array Index)

*File: `portfolio.py*`
Class này lưu trữ:

* `cash_available` và `shares_available`.
* Queue cho Hàng/Tiền chờ về: `pending_cash = [{'amount': X, 'arrival_index': i + 2}]`

**Hàm `update_settlements(current_index)`:** Được gọi đầu tiên mỗi khi vòng lặp chuyển sang một Index mới. Nếu `current_index >= arrival_index`, chuyển từ `pending` sang `available`.

### Bước 3: Vòng lặp Backtest (Mô phỏng ngày thực)

*File: `backtest.py*`
Vòng lặp `for i in range(len(df)):`

1. **Sáng:** Gọi `portfolio.update_settlements(i)`.
2. **Khớp lệnh (Dùng giá Open):** * Kiểm tra tín hiệu được sinh ra từ ngày `i - 1`.
* Nếu MUA: Thực thi ở giá `Open[i]`. Thêm `shares` vào queue với `arrival = i + 2`.
* Nếu BÁN: Thực thi ở giá `Open[i]`. Thêm `cash` vào queue với `arrival = i + 2`.


3. **Chiều (Đóng cửa):** Cập nhật NAV hiện tại (`cash_available` + `pending_cash` + tổng số cổ phiếu * `Close[i]`).
4. **Tối (Ra quyết định):** Đưa `Close[i]` và các `predictions` của index `i` vào file `strategy.py`. Kiểm tra Stop Loss. Ghi nhận tín hiệu chờ để sáng mai (index `i + 1`) thực thi.

### Bước 4: Tính toán Chỉ số (Performance Metrics)

*File: `metrics.py` & `run_backtest.py*`

* Tính Tổng lợi nhuận (`Total Return`).
* Tính `Max Drawdown` (Sụt giảm tài sản lớn nhất từ đỉnh).
* Tỷ lệ thắng (`Win Rate`).
* Lưu kết quả ra file `backtest_result.csv`.

---

## ✅ 8. Tiêu chí Hoàn thành (Definition of Done)

1. **Về kỹ thuật:** * Lệnh CLI `python run_backtest.py --model LSTM --start 2024-01-01 --end 2024-12-31` chạy thành công không lỗi.
* Tốc độ xử lý 1 năm data (khoảng 250 ngày giao dịch) diễn ra dưới 5 giây nhờ cơ chế Batch Predict.


2. **Về nghiệp vụ (QA Checklist):**
* ✅ Output log chứng minh được luật **T+2.5 thông qua Index**: Cổ phiếu mua ở index `i` chỉ khả dụng ở index `i+2`.
* ✅ Khối lượng giao dịch luôn luôn là bội số của `100` (VD: 1500, 2100). Không có số lẻ.
* ✅ Giá khớp lệnh chứng minh được là giá `Open` của ngày hôm sau, tránh hoàn toàn Lookahead Bias.
* ✅ Lệnh Stop Loss tự động kích hoạt nếu giá trị trung bình sụt giảm quá 7%.
* ✅ Tiền bị trừ chuẩn xác theo công thức có phí `0.15%`.


3. **Về báo cáo:**
* Sinh ra file log ghi nhận chi tiết NAV từng ngày.
* Trích xuất được file csv tổng hợp các lệnh MUA/BÁN.