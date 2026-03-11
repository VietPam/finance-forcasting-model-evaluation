import pickle
import os
import numpy as np
from finance_forecast_research.backtest_engine import BacktestEngine

def main():
    file_path = "finance_forecast_research/predictions/LSTM_evaluate_data.pkl"
    
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file {file_path}. Hãy kiểm tra lại thư mục predict.")
        return

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    y_true = data['y_true'] 
    y_pred = data['y_pred'] 
    model_name = data.get('model_name', 'LSTM')

    print(f"--- Đang chạy Backtest cho mô hình: {model_name} ---")
    
    # Khởi tạo vốn 100 triệu, phí giao dịch 0.1%
    engine = BacktestEngine(initial_capital=100000000, commission=0.001)
    
    # Chạy thực tế (lấy cột t+1 của y_true và y_pred)
    metrics = engine.run_backtest(y_true[:, 0], y_pred)

    print("\nKẾT QUẢ ĐẦU TƯ CHI TIẾT:")
    print("-" * 30)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("-" * 30)

    # Nếu bạn đang dùng Cloud Shell (không có màn hình đồ họa), 
    # lệnh plot_results() có thể gây lỗi. Hãy tạm comment nó lại nếu chạy trên server.
    # engine.plot_results()

if __name__ == "__main__":
    main()