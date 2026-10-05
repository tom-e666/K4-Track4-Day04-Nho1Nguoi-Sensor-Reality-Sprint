# Bản nộp cá nhân — Day 19 · T1 Camera Degradation Health Score

**Họ tên:** Thái Phúc Tiến  
**Mã học viên / MSSV:** 2A2026202873  
**Vai trò trong nhóm:** Thiết kế và hiện thực hóa pipeline thực nghiệm (`src/run_demo.py`), cài đặt các thuật toán đo đa chỉ số camera (Laplacian, Canny edge, Clipping/Saturation) & tính Composite Health Score; tích hợp downstream proxy YOLOv8n; thiết lập CI/CD GitHub Actions tự động hóa benchmark và trích xuất số liệu (`results/metrics.csv`, đồ thị minh chứng).  
**Repository:** https://github.com/tom-e666/vin20kday19aisensor @ commit `dcf85a55aaaa1aaee5945365c3536df4f7342e99`

---

## 1. Problem

- **Nền tảng / tính năng / sensor:** Hệ thống hỗ trợ lái xe nâng cao (ADAS) trên ô tô; tính năng giám sát độ tin cậy và suy thoái chất lượng hình ảnh (Camera Degradation Health Score) trước khi đưa vào mô hình nhận diện vật thể (Object Detection); sử dụng cảm biến **Camera RGB** quan sát phía trước xe.
- **Failure case và điều kiện xuất hiện:** 
  - Khi xe di chuyển trong điều kiện thời tiết hoặc môi trường khắc nghiệt dẫn đến: ống kính bị nhòe mờ quang học nghiêm trọng (Optical Blur do sương mù, bẩn ống kính), chói lóa ngược sáng mạnh (Strong Glare do ánh nắng mặt trời chiếu trực diện), hoặc nhiễu hạt cảm biến/ISP (Impulse Noise).
  - Vấn đề cốt lõi: Downstream AI detector (nhận diện vật thể) vẫn có thể đưa ra các bounding box với độ tự tin (confidence) cao hoặc sinh ra false positive vô lý ngay cả khi dữ liệu cấu trúc ảnh của camera đã bị suy thoái hoàn toàn.
- **Claim:** **[Đo]** Khi độ mờ quang học tăng từ ảnh sạch (Baseline) lên mức mờ cực nặng (Gaussian Blur $k=21$), năng lượng tần số cao đo bằng Laplacian variance sụp đổ từ `3619.17` xuống `3.40` (giảm **99.91%**) và độ giữ biên (edge retention) giảm từ `100%` xuống `4.09%`. Tuy nhiên, chỉ số độ tự tin trung bình của YOLOv8n không hề giảm mà ngược lại còn tăng từ `0.656` lên `0.675` (**+3.0%**, giữ nguyên 6 detections). Vì vậy, độ tin cậy của downstream detector không thể dùng làm thước đo sức khỏe cho camera.

---

## 2. Method

- **Paper/repo đã đọc:**
  - **[Nguồn]** *Dong et al., CVPR 2023*, *Benchmarking Robustness of 3D Object Detection to Common Corruptions*: Đề xuất 27 dạng suy thoái cảm biến trên các bộ dữ liệu KITTI-C, nuScenes-C, Waymo-C; chứng minh mô hình thuần camera cực kỳ nhạy cảm và dễ vỡ trước các biến dạng ảnh.
  - **[Nguồn]** *Nam et al., Vehicles 2025*, *BREMOLA*: Phương pháp đánh giá chất lượng ảnh không cần ảnh mẫu (No-Reference IQA) cho xe tự hành, chứng minh toán tử Laplacian có tương quan xếp hạng cao nhất với mức độ mờ (SROCC = 0.9236).
  - **[Nguồn]** *Xie et al., IEEE TPAMI 2025*, *RoboBEV*: Chỉ ra motion blur có thể làm sụt giảm mạnh khả năng nhận diện dù biểu đồ phân phối điểm ảnh (pixel histogram) toàn cục biến động rất ít.
  - **[Nguồn]** *Yang et al., Sensors 2026*, *Camera Lens Soiling Severity*: Chứng minh việc dùng bộ lọc làm mượt chuỗi thời gian giúp triệt tiêu độ rung giật tín hiệu (Jitter MAD) tới 51.5%.
  - **[Nguồn]** Repo nền tảng: `ultralytics/ultralytics` phiên bản `8.4.172` (chạy mô hình YOLOv8n làm downstream detector proxy).

- **Input → output của phương pháp:**
  ```text
  RGB Camera Frame (Ảnh đường phố)
    → Controlled Degradation (Clean / Gaussian Blur / Glare / Salt & Pepper)
    → Trích xuất đặc trưng vật lý ảnh (Laplacian var, Edge retention, Saturation/Clipping ratio, Brightness, Contrast, Entropy)
    → Tính Composite Health Score (H ∈ [0, 1])
    → Downstream Detector Proxy (YOLOv8n: Box count, Mean confidence)
    → Temporal Persistence Check (H < 0.65 trong 3 frame liên tiếp)
    → Quyết định trạng thái (OK / CAMERA_DEGRADED) & kích hoạt Fallback
  ```

- **Metric và dataset của nguồn:**
  - **[Nguồn]** Paper gốc dùng các tập dữ liệu tự hành quy mô lớn (KITTI-C, nuScenes-C, Waymo-C, WoodScape) đo bằng mAP, NDS, RCE, SROCC.

- **Phần nhóm tái hiện và lý do (Workload phụ trách):**
  - **[Đo]** Xây dựng module thực nghiệm gọn nhẹ (`src/run_demo.py`) mô phỏng triết lý controlled corruption trên ảnh đường phố thực tế (`bus.jpg` từ Ultralytics, SHA-256: `c02019c49...`) và ảnh đồ họa tổng hợp offline (`synthetic_road`).
  - Thiết lập công thức đo đa tín hiệu $H = 0.40 S_{sharp} + 0.15 S_{edge} + 0.10 S_{bright} + 0.10 S_{contrast} + 0.25 Q_{clip}$ để bắt trọn cả 3 hiện tượng: mờ (blur), chói (glare) và nhiễu xung (noise).
  - Tích hợp pipeline CI/CD GitHub Actions (`benchmark.yml`) chạy tự động, commit minh chứng và xuất bản báo cáo tĩnh lên GitHub Pages.

---

## 3. Benchmark

- **Dữ liệu:** **[Đo]** Ảnh đường phố `bus.jpg` (kích thước `810×1080`, SHA-256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`).
- **Baseline và các mức lỗi:**
  - Baseline: Ảnh gốc sạch chuẩn (`0_clean.jpg`).
  - Blur nhẹ: Gaussian blur $k=5$, $\sigma \approx 0.8$ px.
  - Blur vừa: Gaussian blur $k=11$, $\sigma \approx 1.8$ px.
  - Blur nặng: Gaussian blur $k=21$, $\sigma \approx 3.5$ px.
  - Chói lóa mạnh (Strong Glare): Vùng bão hòa cục bộ $r = 30\% \min(H, W)$, độ mờ mask $\sigma = 36$ px.
  - Nhiễu hạt (Salt & Pepper): Tỷ lệ nhiễu $p = 0.025$, seed ngẫu nhiên cố định = 19.
- **Lệnh chạy:**
  ```bash
  python src/run_demo.py --input bus.jpg --source-url "https://ultralytics.com/images/bus.jpg" --yolo --output-dir results
  ```

- **Bảng số liệu thực nghiệm đo được (trích từ `results/metrics.csv`):**

| Điều kiện | Tham số | Laplacian Var | Edge Retention | Highlight Saturation | Health Score ($H$) | Detections (YOLO) | Mean Conf. | Bằng chứng |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **Baseline** | `clean` | **3619.17** | **100.0%** | **1.27%** | **1.000 (OK)** | **6** | **0.656** | `results/0_clean.jpg` |
| Blur nhẹ | $k=5$ | 121.50 | 47.8% | 0.52% | 0.529 (DEGRADED) | 6 | 0.653 | `results/1_blur_k5.jpg` |
| Blur vừa | $k=11$ | 13.04 | 15.2% | 0.35% | 0.465 (DEGRADED) | 6 | 0.662 | `results/2_blur_k11.jpg` |
| **Blur nặng** | $k=21$ | **3.40** | **4.09%** | **0.19%** | **0.444 (DEGRADED)** | **6** | **0.675** | `results/3_blur_k21.jpg` |
| **Chói mạnh** | `strong_glare` | 2789.66 | 87.3% | **20.00%** | **0.636 (DEGRADED)** | **7** | **0.549** | `results/4_strong_glare.jpg` |
| **Nhiễu hạt** | $p=0.025$ | **13968.22** | **139.2%** | 2.49% | **0.629 (DEGRADED)** | 6 | 0.597 | `results/5_salt_pepper.jpg` |

- **Đồ thị minh chứng:**
  - `results/degradation_metrics.png`: Biểu đồ trực quan hóa sụt giảm đặc trưng ảnh qua các mức lỗi.
  - `results/detector_proxy.png`: Biểu đồ đối chứng giữa Health Score và YOLO Confidence/Detections.
  - `results/detector_before_after.png`: Bounding box minh chứng sai lệch nghiêm trọng khi bị glare.

---

## 4. Failure case

- **[Đo] Nhóm quan sát được từ số liệu:**
  1. *Optical Blur ($k=21$):* Dù Laplacian variance giảm **99.91%** và biên ảnh gần như biến mất hoàn toàn, YOLOv8n vẫn trả ra nguyên vẹn 6 boxes với mean confidence tăng từ `0.656` lên `0.675` (+3.0%). Điều này chứng minh detector bị mù thông tin cấp thấp nhưng vẫn tự tin sai lệch.
  2. *Strong Glare (Chói sáng):* Pixel cháy sáng bão hòa vọt lên `20.00%` (gấp 15.7 lần baseline), làm rớt độ tự tin trung bình xuống `0.549` nhưng số lượng detection lại tăng từ 6 lên 7 boxes. Đáng chú ý, overlay detector xuất hiện một box false-positive vô lý: nhãn `airplane 0.45` đè lên phần thân xe bus bị lóa sáng.
  3. *Salt & Pepper (Nhiễu hạt):* Tạo ra các cạnh sắc giả tạo khiến Laplacian variance tăng vọt lên `13968.22` (gấp ~3.86 lần ảnh sạch) và Edge retention đạt `139.2%`. Nếu chỉ dùng luật đơn giản "Laplacian cao = ảnh tốt" thì hệ thống sẽ nhận định sai lầm rằng camera đang rất nét.

- **[Nguồn] Paper/repo cho biết:**
  - BREMOLA xác nhận Laplacian bắt blur rất tốt (SROCC 0.9236) nhưng không thể xử lý đơn lẻ cho glare và noise.
  - RoboBEV chỉ ra perception BEV sụt giảm nghiêm trọng khi camera lỗi dù phân phối histogram toàn cục ít thay đổi.

- **[Giả thuyết] Nguyên nhân / tác động tới tính năng:**
  - Mean confidence của YOLO chỉ tính trên tập con các box vượt qua ngưỡng threshold NMS; khi các box mờ bị triệt tiêu, box còn lại vô tình có điểm trung bình cao hơn.
  - Trên xe ADAS thực tế, hiện tượng này có thể dẫn tới việc xe không phát hiện chướng ngại vật phía trước (False Negative) hoặc tự kích hoạt phanh gấp khẩn cấp AEB vì nhận diện nhầm bóng lóa (Ghost Braking).

- **Limitation của phép thử:**
  - Phép thử thực hiện trên **1 ảnh tĩnh** mẫu; chưa đo được mAP/Recall chuẩn do thiếu nhãn ground-truth annotation.
  - Trọng số $H$ mang tính heuristic, cần được hiệu chuẩn trên tập video xe chạy thực tế có nhiều điều kiện thời tiết (ngày, đêm, mưa).

---

## 5. Engineering decision

- **Cải tiến / fallback đề xuất:**
  1. **Tách biệt độc lập:** Không sử dụng chỉ số confidence hay detection count của mô hình nhận diện làm tín hiệu đo sức khỏe camera; bắt buộc đặt bộ lọc Camera Health Monitor ở tầng tiền xử lý (pre-perception).
  2. **Đa tín hiệu tương hỗ (Multi-signal Fusion):** Kết hợp đồng thời Laplacian (bắt blur), Clipping/Saturation ratio (bắt chói lóa/tối om), và Canny symmetric retention (trừ điểm khi biên tăng đột biến do nhiễu hạt).
  3. **Bộ lọc trễ thời gian (Temporal Persistence Rule):** Chỉ kích hoạt sự kiện `CAMERA_DEGRADED` khi $H < 0.65$ duy trì liên tiếp trong **ít nhất 3 khung hình** (tránh kích hoạt nhầm khi xe chỉ đi qua đèn đường hoặc tia sét chớp trong 1 frame).
  4. **Cơ chế Fallback ADAS:** Khi camera rơi vào trạng thái degraded $\rightarrow$ Ngay lập tức gửi tín hiệu hạ trọng số camera trong bộ hợp nhất đa cảm biến (Sensor Fusion), chuyển quyền quan sát ưu tiên sang Radar sóng milimet / LiDAR, đồng thời phát tín hiệu cảnh báo tài xế tiếp quản lái hoặc kích hoạt chế độ giảm tốc an toàn (Safe Stop).

- **Metric / log dùng để kiểm chứng ở vòng sau, và tiêu chí đạt:**
  - Log chi tiết từng frame: Timestamp, Camera Gain/ISO, Exposure time, Shutter, Clipping ratio, Laplacian variance, Composite Health score.
  - Tiêu chí đạt: Tỷ lệ báo động giả (False Alarm Rate) trên chuỗi video sạch $\le 5\%$; độ trễ phát hiện suy thoái $\le 100$ ms; nhận diện thành công $\ge 95\%$ các trường hợp chói sáng hoặc nhòe mờ thực tế.

- **Khi nào nên / không nên dùng phương án này (Trade-off):**
  - *Nên dùng:* Các hệ thống ADAS thương mại cần giải pháp giám sát cảm biến siêu nhẹ (lightweight, chạy realtime ở tần số 30–60 FPS trên chip nhúng mà không tốn GPU).
  - *Không nên dùng đơn lẻ:* Trong các hệ thống tự hành cấp độ cao (L4/L5) khi chưa tích hợp phân tích vùng không gian cụ thể (spatial/ROI-aware); lúc này cần kết hợp thêm mạng nơ-ron học sâu để phát hiện chính xác vật cản bám dính cục bộ trên ống kính (soiling segmentation).