# 📈 Backtesting Engine - Implementation Plan (VN Market)

## 🎯 1. Mục tiêu (Goal & Scope)

Xây dựng một hệ thống Backtest mô phỏng giao dịch cổ phiếu (ví dụ: ACB) dựa trên dự báo của mô hình AI (DEFAULT_ANN hoặc LSTM). Hệ thống sẽ "quay ngược thời gian", duyệt qua từng ngày trong quá khứ, đưa ra quyết định Mua/Bán/Nắm giữ dựa trên dự đoán của AI, đồng thời tuân thủ nghiêm ngặt các quy định thực tế của thị trường chứng khoán Việt Nam (HOSE).

**Phạm vi (Scope):**

* **Input:** Tập dữ liệu lịch sử OHLCV (Open, High, Low, Close, Volume) và các mô hình AI đã được train.
* **Output:** Báo cáo hiệu suất đầu tư (Return, Drawdown, Win Rate) và chi tiết lịch sử giao dịch.
* **Giới hạn hiện tại:** Chỉ chạy trên khung thời gian Ngày (Daily Chart). Không margin (vay nợ), không bán khống (short selling).

---

## 📚 2. Định nghĩa & Luật chơi thực tế (Market Rules)

Để code chạy sát với thực tế, team Backend cần nắm rõ các rule sau (Hard-coded vào hệ thống):

### 2.1. Quy tắc Thanh toán T+2.5 (Cổ phiếu & Tiền)

Ở Việt Nam, giao dịch không hoàn tất ngay lập tức. Hệ thống phải theo dõi "trạng thái chờ" của Cổ phiếu và Tiền.

* **Chu kỳ Cổ phiếu (Mua):**
* Sáng Thứ 2 (Ngày T): Đặt lệnh MUA khớp thành công. Tiền bị trừ ngay. Cổ phiếu ở trạng thái "chờ về".
* Trưa Thứ 4 (Ngày T+2): Cổ phiếu chính thức về tài khoản.
* **Chiều Thứ 4 (Ngày T+2): Bắt đầu được phép đặt lệnh BÁN số cổ phiếu này.**


* **Chu kỳ Tiền (Bán):**
* Chiều Thứ 4 (Ngày T): Đặt lệnh BÁN khớp thành công. Cổ phiếu bị trừ ngay. Tiền ở trạng thái "chờ về".
* Trưa Thứ 6 (Ngày T+2): Tiền chính thức về tài khoản.
* **Chiều Thứ 6 (Ngày T+2): Có thể dùng số tiền này để MUA tiếp.** *(Lưu ý: Bỏ qua nghiệp vụ "ứng trước tiền bán" để backtest an toàn).*



**👉 Suy ra Logic Code (Dựa trên index mảng `i` đại diện cho Ngày):**

* Lệnh MUA khớp ở index `i` $\rightarrow$ Đến index `i+2` (sau 2 ngày giao dịch) mới cập nhật số dư Cổ phiếu khả dụng để Bán.
* Lệnh BÁN khớp ở index `j` $\rightarrow$ Đến index `j+2` mới cập nhật số dư Tiền mặt khả dụng để Mua.

### 2.2. Khối lượng giao dịch (Lot Size)

Sàn HOSE quy định chỉ được phép giao dịch bội số của 100 cổ phiếu (lô chẵn).

* **Công thức tính số cổ phiếu tối đa có thể mua:**
```python
# 1. Tính số lượng lý thuyết
raw_shares = available_cash / (price * (1 + transaction_fee))
# 2. Làm tròn xuống bội số của 100
actual_shares = (int(raw_shares) // 100) * 100

```



### 2.3. Chi phí giao dịch (Transaction Cost)

* **Phí quy định:** **0.15%** cho mỗi chiều Mua và chiều Bán (Đã bao gồm thuế).
* Ví dụ: Mua 100 triệu tiền cổ phiếu $\rightarrow$ Mất 150.000đ tiền phí. Tài khoản bị trừ 100.150.000đ. Bán 100 triệu cổ phiếu $\rightarrow$ Thu về 99.850.000đ.

### 2.4. Tránh bẫy Lookahead Bias (Rất quan trọng)

Mô hình AI nhận input là giá Đóng cửa (Close) của ngày $T$ để đưa ra dự báo. Do đó, tín hiệu giao dịch chỉ có **SAU KHI** ngày $T$ kết thúc.

* **Sai lầm:** Có tín hiệu vào tối ngày $T$, lùi thời gian lại lấy giá Close ngày $T$ để mua/bán $\rightarrow$ Lãi ảo.
* **Logic đúng:** Tín hiệu sinh ra từ cuối ngày $T$ $\rightarrow$ Lệnh sẽ được thực thi vào **Ngày $T+1$**. Giá khớp lệnh sẽ là giá `Open` (Mở cửa) hoặc `Close` (Đóng cửa) của ngày $T+1$.

---

## 🎮 3. Chiến lược giao dịch (Trading Strategy 1 - Conservative)

Mô hình dự báo giá của 3 ngày tiếp theo: `pred_t1`, `pred_t2`, `pred_t3`.

**Điều kiện sinh tín hiệu từ ngày T (để hành động vào ngày T+1):**

1. **Tín hiệu MUA (BUY):**
* Cả 3 dự báo đều cao hơn giá Đóng cửa hiện tại: `pred_t1 > Close_T` AND `pred_t2 > Close_T` AND `pred_t3 > Close_T`.
* **Hành động:** Nạp lệnh Mua toàn bộ bằng Tiền mặt khả dụng vào ngày $T+1$.


2. **Tín hiệu BÁN (SELL):**
* Có ít nhất 2 trong 3 dự báo thấp hơn giá Đóng cửa hiện tại (Đa số vote Giảm).
* **Hành động:** Nếu có Cổ phiếu khả dụng (đã qua T+2.5), nạp lệnh Bán toàn bộ vào ngày $T+1$.


3. **ĐỨNG NGOÀI (HOLD):**
* Tín hiệu nhiễu (không thỏa Mua/Bán), hoặc đang chờ Cổ phiếu/Tiền về.



---

## 🏗️ 4. Kiến trúc Hệ thống (System Architecture)

```text
finance_forecast_backtest_engine/
├── __init__.py
├── data_loader.py       # Load CSV, xử lý features, batch_predict để tối ưu hiệu năng
├── portfolio.py         # Quản lý Cash, Shares, trạng thái T+2 (Tiền chờ về, Hàng chờ về)
├── strategy.py          # Implement logic Chiến lược 1 (3 ngày UP/DOWN)
├── backtest.py          # Vòng lặp thời gian chính (Time-loop simulation)
└── metrics.py           # Tính toán Sharpe, Drawdown, Return, Win Rate

run_backtest.py          # File chạy chính qua CLI

```

---

## 📋 5. Các bước Thực thi Chi tiết (Implementation Steps)

### Bước 1: Xử lý Dữ liệu & Dự báo hàng loạt (Tối ưu hiệu năng)

*File: `data_loader.py*`

* **Không dùng:** Chạy `model.predict()` bên trong vòng lặp từng ngày (Rất chậm).
* **Cách làm đúng:** Quét qua data, tạo một mảng Numpy 3D chứa toàn bộ cửa sổ 30 ngày của tập Test. Gọi `predictions = model.predict(all_features)` **1 lần duy nhất**. Lưu kết quả này vào một DataFrame có cột `Date` để map với giá thực tế.

### Bước 2: Xây dựng Portfolio Manager

*File: `portfolio.py*`
Class này là trái tim của hệ thống. Nó cần các thuộc tính:

* `cash_available`: Tiền có thể dùng ngay.
* `shares_available`: Cổ phiếu có thể bán ngay.
* `pending_cash`: List các khoản tiền đang chờ về `[{'amount': 10tr, 'arrival_day_index': 15}]`
* `pending_shares`: List các lô cổ phiếu chờ về `[{'amount': 1000, 'arrival_day_index': 15}]`

**Hàm quan trọng:** `update_settlements(current_day_index)`: Chạy đầu mỗi ngày để chuyển Tiền/Cổ phiếu từ `pending` sang `available` nếu `current_day_index >= arrival_day_index`.

### Bước 3: Viết vòng lặp Backtest

*File: `backtest.py*`
Luồng chạy cho mỗi ngày `i`:

1. **Sáng (Bắt đầu ngày):** Gọi `portfolio.update_settlements(i)`. Cập nhật trạng thái Tiền/Hàng.
2. **Thực thi lệnh Tồn đọng (Mở cửa):** Kiểm tra xem tối qua (ngày `i-1`) có sinh ra tín hiệu Mua/Bán nào không. Nếu có, thực thi lệnh ở giá `Close[i]` (hoặc `Open[i]`).
* *Mua:* Trừ `cash_available`, thêm vào `pending_shares` (với `arrival = i + 2`).
* *Bán:* Trừ `shares_available`, thêm vào `pending_cash` (với `arrival = i + 2`).


3. **Cuối ngày:** Tính toán tổng tài sản (NAV = Tiền khả dụng + Tiền chờ về + Hàng khả dụng * giá + Hàng chờ về * giá). Lưu vào mảng lịch sử.
4. **Phân tích & Ra quyết định cho ngày mai:** Lấy dự báo `pred_t1, t2, t3` sinh ra từ ngày `i`. Chạy qua file `strategy.py`. Nếu có tín hiệu, lưu lệnh vào hàng đợi để sáng mai (ngày `i+1`) thực thi.

### Bước 4: Tính toán Chỉ số & Báo cáo

*File: `metrics.py` & `run_backtest.py*`

* Tính `Total Return (%)`.
* Tính `Max Drawdown` (Giúp biết lúc xui xẻo nhất tài khoản bị chia bao nhiêu %).
* Xuất ra CSV lịch sử giao dịch và biểu đồ đường cong vốn (Equity Curve).

---

## ✅ 6. Tiêu chí Hoàn thành (Definition of Done)

1. **Về kỹ thuật:** Lệnh CLI `python run_backtest.py --model LSTM --start 2024-01-01 --end 2024-12-31` chạy thành công không lỗi, tốc độ xử lý 1 năm data dưới 5 giây.
2. **Về nghiệp vụ (Kiểm thử QA):**
* ✅ Output log chứng minh được luật **T+2.5**: Không có trường hợp nào Mua xong bán được ngay trong hôm sau. Tiền bán xong không được dùng để mua ngay ngày mai.
* ✅ Khối lượng giao dịch luôn luôn kết thúc bằng hai số 0 (Ví dụ: 1500, 2100). Không có số lẻ (Ví dụ: 1523).
* ✅ Tiền bị trừ khớp đúng công thức: `Khối lượng * Giá * 1.0015`.
* ✅ Lệnh được khớp vào ngày $T+1$ (để giải quyết Lookahead Bias).


3. **Về báo cáo:** Sinh ra file `backtest_result.csv` ghi nhận chi tiết NAV từng ngày và lịch sử các lệnh đã đặt.

---

**Sẵn sàng triển khai! Hãy bắt đầu từ việc setup cấu trúc thư mục và tạo module `data_loader.py` theo đúng tinh thần "Dự báo Batch" để tối ưu hiệu năng.** 🚀