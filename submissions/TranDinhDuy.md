# Bản nộp cá nhân — Day 19 · T1 Camera Degradation Health Score

> **Quy định đối chiếu:** Số liệu thực nghiệm được trích xuất trực tiếp từ các file bằng chứng chung của nhóm trong repository: `results/metrics.csv`, `results/summary.json`, `results/policy_sequence.csv`, `REPORT.md`, `PITCH.md`.  
> Phân định minh bạch ba loại khẳng định: **[Đo]** = nhóm tự đo đạc trong benchmark · **[Nguồn]** = paper/repo nghiên cứu công bố · **[Giả thuyết]** = nhận định kỹ thuật cần kiểm chứng thêm ở hệ thống hoàn chỉnh.

**Họ tên:** Trần Đình Duy  
**MSSV:** 2A202602631  
**Vai trò trong nhóm:** Phân tích các ca thất bại và đề xuất chiến lược kỹ thuật (Failure Analysis & Engineering Decisions, Pitching): trực tiếp phân tích các ca gãy của mô hình thị giác máy tính phía sau (YOLOv8n bị tăng confidence khi ảnh mờ, chói sáng sinh nhãn máy bay ảo, nhiễu hạt đánh lừa bộ lọc tần số cao); xây dựng & mô phỏng chính sách lọc trễ chuỗi thời gian (3-frame temporal persistence policy); đề xuất cơ chế an toàn dự phòng (fallback) và danh mục telemetry cho xe ADAS (chịu trách nhiệm chính mục §4, §5 trong `REPORT.md` và `PITCH.md`).  
**Repository:** https://github.com/tom-e666/K4-Track4-Day04-Nho1Nguoi-Sensor-Reality-Sprint @ commit `e477ec600385a18e906fb176a7ec271af618b04b`  
*(Tên thư mục quy định: `K4-Track4-Day04-Nho1Nguoi-Sensor-Reality-Sprint`)*

---

## 1. Problem

- **Nền tảng / tính năng / sensor:**
  - Nền tảng: Hệ thống hỗ trợ lái xe nâng cao (ADAS Level 2+ / Level 3) trên ô tô.
  - Tính năng: Giám sát độ tin cậy và suy thoái chất lượng cảm biến (Camera Degradation Health Score) trước khâu nhận diện vật thể (Object Detection) cấp dữ liệu cho Phanh khẩn cấp tự động (AEB) và Kiểm soát hành trình thích ứng (ACC).
  - Cảm biến: Camera đơn sắc thái RGB góc rộng hướng thẳng phía trước xe (Forward-facing Monocular Camera).

- **Failure case và điều kiện xuất hiện:**
  - Trong thực tế vận hành, camera thường xuyên gặp các điều kiện quang học khắc nghiệt:
    1. Nhòe mờ quang học nghiêm trọng (Optical Blur / Motion Blur do sương mù, mưa phùn, thấu kính bị bám hơi ẩm hoặc xe chạy rung lắc tốc độ cao).
    2. Chói lóa cục bộ cường độ mạnh (Local Glare / Over-exposure do ánh nắng mặt trời chiếu trực diện bình minh/hoàng hôn hoặc đèn pha pha thẳng vào ống kính).
    3. Nhiễu hạt cảm biến/ISP (Impulse Noise / Salt-and-Pepper do cảm biến quá nhiệt, thiếu sáng hoặc lỗi truyền dữ liệu bus).
  - **Nghịch lý nguy hiểm cốt lõi:** Các mô hình phát hiện vật thể học sâu (Deep Learning Object Detector như YOLO) vẫn đưa ra các bounding box với độ tự tin (Confidence) cao hoặc giữ nguyên số lượng đối tượng dự đoán ngay cả khi khung hình đã bị hủy hoại chi tiết cấu trúc nghiêm trọng. Nếu xe chỉ dựa vào detector confidence, hệ thống sẽ bị "mù" trước các hỏng hóc cảm biến.

- **Claim (với đơn vị đo lường cụ thể):**
  - **[Đo]** Khi độ mờ Gaussian blur tăng từ ảnh sạch (`clean`) lên mức cực nặng (`blur_gauss_k21`, $\sigma = 3.5\text{ px}$):
    - Laplacian variance sụp đổ từ $3619.17$ xuống còn $3.40\text{ px-intensity}^2$ (giảm **99.91%**).
    - Tỷ lệ giữ biên (edge retention) rơi từ $100.0\%$ xuống chỉ còn $4.09\%$.
    - Điểm sức khỏe ảnh tổng hợp tụt xuống $0.444$ (rơi sâu vào ngưỡng `DEGRADED`).
    - **Tuy nhiên:** Độ tự tin trung bình (mean confidence) của mô hình YOLOv8n **không hề giảm mà ngược lại còn tăng +3.0%** (từ $0.656$ lên $0.675$), và số lượng detection giữ nguyên $6 \to 6$.
  - $\Rightarrow$ Do đó, độ tin cậy của downstream detector hoàn toàn không thể thay thế cho một bộ giám sát sức khỏe cảm biến độc lập phía đầu vào (pre-perception sensor health monitor).

---

## 2. Method

- **Paper / repo đã đọc và đối chiếu:**
  - **[Nguồn]** *Dong et al., CVPR 2023*, “*Benchmarking Robustness of 3D Object Detection to Common Corruptions*”: Xây dựng bộ chuẩn KITTI-C, nuScenes-C, Waymo-C với 27 loại suy thoái ở 5 mức độ; chứng minh mô hình thuần camera (camera-only) dễ bị tổn thương nhất trước các biến dạng hình ảnh so với hệ đa cảm biến (LiDAR-Camera fusion).
  - **[Nguồn]** *Xie et al., IEEE TPAMI 2025*, “*RoboBEV: Benchmarking and Improving Bird’s Eye View Perception Robustness in Autonomous Driving*”: Khảo sát 8 loại lỗi camera trên 33 mô hình BEV; chỉ ra hiện tượng motion blur làm suy giảm trầm trọng khả năng nhận thức 3D dù phân phối điểm ảnh toàn cục thay đổi rất ít.
  - **[Nguồn]** *Nam et al., Vehicles 2025*, “*BREMOLA: No-Reference Image Quality Assessment with Moving Spectrum and Laplacian Filter for Autonomous Driving Environment*”: Đề xuất thuật toán đánh giá độ mờ không cần ảnh mẫu (No-Reference) kết hợp phổ chuyển động Fourier và độ phức tạp Laplacian. Báo cáo Laplacian filter đạt tương quan Spearman cao nhất (SROCC = 0.9236). Nhóm đã tích hợp và chạy trực tiếp code upstream tại commit `7ba26999c265692bb8e44c5a3f2d91c06746830f`.
  - **[Nguồn]** *Yang et al., Sensors 2026*, “*A Static-to-Temporal Framework for Interpretable Camera Lens Soiling Severity Estimation*”: Chứng minh bộ lọc chuỗi thời gian (Adaptive EMA) giúp triệt tiêu độ rung giật tín hiệu (Jitter MAD) tới 51.5% khi đánh giá độ bẩn ống kính.

- **Input $\to$ Output của phương pháp:**
  ```text
  Khung hình RGB camera (810 × 1080)
    → 9 biến thể suy thoái có kiểm soát (Blur / Glare / Noise)
    → Trích xuất đa chỉ số ảnh: Laplacian var, Canny edge retention, Clipping/Saturation, Brightness/Contrast, Entropy
    → Tính Composite Health Score (H) [kết hợp hàm đối xứng symmetric ratio similarity]
    → Chạy song song upstream BREMOLA (điểm 0–100) & downstream YOLOv8n (conf, count, pseudo-label agreement)
    → Đưa qua bộ lọc trễ chuỗi thời gian (3-Frame Persistence Policy)
    → Trạng thái sức khỏe cảm biến (OK / DEGRADED) & Kích hoạt cơ chế Fallback
  ```

- **Metric và dataset của nguồn:**
  - **[Nguồn]** Dong et al. & RoboBEV: Đánh giá bằng mAP, NDS, RCE (Relative Corruption Error), mCE trên hàng chục nghìn frame nuScenes/Waymo.
  - **[Nguồn]** BREMOLA: Đo trên tập 45 khung hình thực tế kèm 20 mức độ mờ (tổng 900 ảnh), dùng hệ số tương quan SROCC/PLCC.
  - **[Nguồn]** Yang et al.: Đánh giá độ bẩn ống kính bằng Spearman correlation và độ ổn định thời gian Jitter (MAD).

- **Phần nhóm tái hiện và lý do:**
  - **[Đo]** Nhóm không chạy lại toàn bộ tập dữ liệu khổng lồ của KITTI-C/RoboBEV vì thời lượng lab và tài nguyên tính toán giới hạn. Nhóm tái hiện phương pháp luận *controlled corruption*: giữ nguyên ảnh gốc chuẩn làm mốc đối chứng và biến đổi từng tham số độc lập với mức độ tăng dần.
  - Với BREMOLA: Nhóm nhúng trực tiếp mã nguồn Python gốc của tác giả (chạy trên CPU không chỉnh sửa) để đối chiếu độc lập với Composite Health Score của nhóm.

---

## 3. Benchmark

- **Dữ liệu thực nghiệm:**
  - Ảnh giao thông đường phố thực tế từ Ultralytics (`bus.jpg`, kích thước $810 \times 1080$).
  - SHA-256 mã băm: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.
- **Baseline và các mức lỗi:**
  - Gồm 1 mốc chuẩn (`clean`) và 9 biến thể lỗi thuộc 3 nhóm chính:
    1. Gaussian Blur: $k=5$ ($\sigma=0.83$), $k=11$ ($\sigma=1.83$), $k=21$ ($\sigma=3.50\text{ px}$).
    2. Local Glare: Gain độ chói $0.35$, $0.70$, $1.05$ (bán kính $243\text{ px}$, tâm lóa $(788, 162)$).
    3. Salt-and-Pepper Noise: Tỷ lệ hạt $p=0.005$, $p=0.010$, $p=0.025$ (seed 19).
- **Lệnh chạy tái hiện:**
  ```bash
  python src/run_demo.py \
    --input assets/bus.jpg \
    --source-url https://ultralytics.com/images/bus.jpg \
    --yolo \
    --bremola-script third_party/BREMOLA/bremola.py \
    --output-dir results
  ```

- **Bảng số liệu thực nghiệm chính (Trích xuất từ [`results/metrics.csv`](results/metrics.csv)):**

| Điều kiện | Tham số lỗi | Laplacian var. | Highlight Sat. | BREMOLA | Health Score | State | Detections | Mean Conf. | Agreement | Bằng chứng |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Baseline** | `clean` | 3619.17 | 1.27% | 38.91 | **1.000** | OK | 6 | 0.656 | 100% | [`results/00_clean.jpg`](results/00_clean.jpg) |
| Blur nhẹ | $k=5, \sigma=0.83$ | 121.50 | 0.52% | 29.30 | **0.529** | DEGRADED | 6 | 0.653 | 100% | [`results/01_blur_gauss_k5.jpg`](results/01_blur_gauss_k5.jpg) |
| Blur vừa | $k=11, \sigma=1.83$ | 13.04 | 0.35% | 29.58 | **0.465** | DEGRADED | 6 | 0.662 | 83.3% | [`results/02_blur_gauss_k11.jpg`](results/02_blur_gauss_k11.jpg) |
| **Blur nặng** | $k=21, \sigma=3.50$ | 3.40 | 0.19% | 28.56 | **0.444** | **DEGRADED** | 6 | **0.675** | 83.3% | [`results/detector_blur_compare.png`](results/detector_blur_compare.png) |
| Glare nhẹ | gain $= 0.35$ | 3500.08 | 4.97% | 39.12 | **0.940** | OK | 6 | 0.655 | 100% | [`results/04_glare_gain0.35.jpg`](results/04_glare_gain0.35.jpg) |
| Glare vừa | gain $= 0.70$ | 3129.94 | 11.42% | 39.42 | **0.803** | OK | 6 | 0.643 | 100% | [`results/05_glare_gain0.70.jpg`](results/05_glare_gain0.70.jpg) |
| **Glare mạnh**| gain $= 1.05$ | 2789.66 | 20.00% | 40.39 | **0.636** | **DEGRADED** | 7 | **0.549** | 83.3% | [`results/detector_before_after.png`](results/detector_before_after.png) |
| Noise nhẹ | $p = 0.005$ | 5736.40 | 1.52% | 41.03 | **0.832** | OK | 5 | **0.741** | 83.3% | [`results/07_saltpepper_p0.005.jpg`](results/07_saltpepper_p0.005.jpg) |
| Noise vừa | $p = 0.010$ | 7809.65 | 1.76% | 41.41 | **0.749** | OK | 5 | 0.719 | 83.3% | [`results/08_saltpepper_p0.010.jpg`](results/08_saltpepper_p0.010.jpg) |
| **Noise mạnh**| $p = 0.025$ | 13968.22 | 2.49% | 40.53 | **0.629** | **DEGRADED** | 6 | 0.597 | 83.3% | [`results/09_saltpepper_p0.025.jpg`](results/09_saltpepper_p0.025.jpg) |

---

## 4. Failure Case

### 4.1. Chi tiết phân tích ca thất bại từ thực nghiệm
- **[Đo] Ca 1 — Blur k=21: "Detector mù nhưng lại tự tin hơn":**
  - Mất mát cấu trúc: Laplacian variance giảm từ $3619.17 \to 3.40$ (mất **99.91%**), edge retention chỉ còn **4.09%**.
  - Hành vi nghịch lý của YOLOv8n: Số lượng bounding box giữ nguyên là 6, nhưng mean confidence lại **tăng từ 0.656 lên 0.675** (+3.0%).
  - Mổ xẻ nguyên nhân: Bằng chứng tại [`results/detector_blur_compare.png`](results/detector_blur_compare.png) cho thấy đối tượng `stop sign 0.26` (độ tin cậy thấp) bị nhòe mất, nhưng mô hình lại phát hiện nhầm một box sai nghiêm trọng là `dog 0.36` (độ tin cậy cao hơn). Việc loại bỏ box thấp và thêm box sai tự tin cao đã kéo trung bình cộng confidence tăng lên, che giấu hoàn toàn sự cố camera.
- **[Đo] Ca 2 — Glare gain 1.05: "Chói sáng tạo đối tượng ảo (Hallucination)":**
  - Tỷ lệ pixel bão hòa trắng chói (highlight saturation $\ge 245$) vọt từ $1.27\% \to 20.00\%$ (gấp **15.7 lần**), tổng pixel bị xén biên (clipped pixels) tăng lên $23.52\%$.
  - Hành vi của YOLOv8n: Bằng chứng tại [`results/detector_before_after.png`](results/detector_before_after.png) cho thấy đối tượng chủ đạo của khung hình là `bus 0.87` đã hoàn toàn biến mất. Thay vào đó, mô hình sinh ra 2 box ảo vô lý: **`airplane 0.45`** và **`truck 0.29`** đè lên vùng lóa sáng của xe buýt! Tổng số detection tăng từ 6 lên 7.
- **[Đo] Ca 3 — Impulse Noise: "Bẫy năng lượng tần số cao đánh lừa bộ đo sắc nét đơn thuần":**
  - Tại mức $p=0.025$, các hạt nhiễu xung làm Laplacian variance tăng vọt lên $13968.22$ (gấp **3.86 lần** ảnh sạch), edge density tăng tới $139.2\%$.
  - Nếu áp dụng quy tắc trực giác ngây thơ *"Laplacian càng cao thì camera càng nét"*, thuật toán sẽ kết luận camera "siêu khỏe". Nhưng nhờ thiết kế **hàm đối xứng symmetric similarity** $\min(\text{val}/\text{base}, \text{base}/\text{val})$, hệ thống đã phạt nặng năng lượng nhân tạo này, kéo Health Score rơi đúng về $0.629$ (`DEGRADED`).
  - Đáng chú ý ở mức nhiễu nhẹ $p=0.005$, YOLO mất 1 box thấp khiến mean confidence vọt lên $0.741$ (+13.1%).

### 4.2. Đối chiếu với công bố khoa học (Literature Cross-Check)
- **[Nguồn] Giới hạn của BREMOLA:**
  - BREMOLA (Nam et al. 2025) phản ứng rất tốt với Blur (điểm giảm từ 38.91 xuống 28.56 ở $k=21$). Tuy nhiên, khi gặp Chói sáng (Glare) và Nhiễu (Noise), điểm BREMOLA lại **tăng ngược lên 40.39 – 41.41**.
  - Chính các tác giả Nam et al. trong bài báo cũng thừa nhận BREMOLA chỉ thiết kế chuyên biệt cho blur và không bao quát glare/noise. Điều này chứng minh một bộ giám sát an toàn xe tự hành **bắt buộc phải kết hợp đa tín hiệu** (Multi-signal Health Score), không thể chỉ tin vào một thuật toán duy nhất.
- **[Nguồn] RoboBEV (Xie et al. 2025):** Xác nhận rằng các mạng nơ-ron nhận thức phía sau có độ suy thoái không tuyến tính đối với biến dạng hình ảnh; các vùng mất mát thông tin cục bộ (như glare) có thể làm phá vỡ biểu diễn đặc trưng không gian.

### 4.3. Giả thuyết kỹ thuật và tác động tới an toàn xe ADAS
- **[Giả thuyết] Hậu quả trên hệ thống điều khiển thực tế:**
  - Trong kịch bản xe chạy trên cao tốc với tính năng Adaptive Cruise Control (ACC) hoặc Phanh khẩn cấp (AEB): xe buýt phía trước đóng vai trò là "Lead Vehicle". Khi gặp ánh nắng ngược (glare gain 1.05), việc detector đột ngột mất nhãn `bus` và nhảy sang nhãn `airplane` sẽ làm bộ lọc bám vết (Object Tracker) bị vỡ ID track (ID switch/track drop). Hậu quả: hệ thống AEB có thể hiểu nhầm chướng ngại vật đã biến mất hoặc coi là vật thể trên trời, dẫn tới không phanh kịp thời và gây tai nạn đâm va đuôi xe.
- **Limitation của phép thử hiện tại:**
  - Thực nghiệm mới thực hiện trên 1 khung hình tĩnh (`bus.jpg`) kèm nhiễu nhân tạo, chưa phải video chạy liên tục trên đường với thời tiết thực.
  - Điểm Health hiện tại mang tính tham chiếu tương đối (so với ảnh clean gốc). Trong thực tế xe chạy, không có ảnh clean tham chiếu trực tiếp mà cần tính toán theo baseline thích nghi (rolling adaptive baseline) hoặc phối hợp tham số phần cứng ISP.

---

## 5. Engineering Decision

### 5.1. Chính sách lọc trễ chuỗi thời gian (3-Frame Persistence Policy)
- **Nguyên lý:** Một khung hình đơn lẻ bị giảm điểm sức khỏe có thể chỉ do hiện tượng nhất thời (đèn đường nhấp nháy, xe chạy thoáng qua bóng râm, tia sét). Nếu báo động ngay sẽ gây nhiễu loạn điều khiển (chattering).
- **Quy tắc:**  
  $$\text{Chỉ kích hoạt trạng thái } \mathbf{CAMERA\_DEGRADED} \iff H < 0.65 \text{ duy trì liên tiếp qua } \mathbf{\ge 3 \text{ khung hình}}.$$
- **[Đo] Bằng chứng mô phỏng chuỗi 23 khung hình ([`results/policy_sequence.csv`](results/policy_sequence.csv), [`results/policy_timeline.png`](results/policy_timeline.png)):**
  - Khung 5 (glare thoáng qua 1 frame) và khung 10–11 (noise thoáng qua 2 frames): Bộ lọc đơn khung hình (single-frame threshold) phát ra **3 cảnh báo sai (False Alarms)**; trong khi quy tắc **3-Frame Persistence dập tắt hoàn toàn về 0 cảnh báo sai**.
  - Khung 15–22 (suy thoái blur kéo dài): Chính sách kích hoạt cảnh báo chính xác tại khung 17 (độ trễ phản ứng đúng 2 frames $\approx 66\text{ ms}$ ở tốc độ 30 fps — hoàn toàn nằm trong ngân sách an toàn thời gian thực của xe).

### 5.2. Đề xuất quy trình ứng phó dự phòng (Fallback Recommendation)
Khi trạng thái `CAMERA_DEGRADED` được xác nhận:
1. **Phát sự kiện Telemetry:** Ghi log chẩn đoán cảm biến tức thời vào hộp đen hệ thống.
2. **Điều chỉnh bộ Hợp nhất cảm biến (Confidence-Aware Sensor Fusion):** Lập tức **hạ trọng số tin cậy (down-weight) hoặc ngắt kênh hình ảnh camera** khỏi luồng xử lý nhận thức; nâng quyền hạn quyết định lên các cảm biến không bị ảnh hưởng bởi ánh sáng quang học như **Radar sóng milimet hoặc LiDAR**.
3. **Kích hoạt chế độ vận hành thoái lui an toàn (Degraded Operational Mode):** 
   - Tự động ngắt tính năng tự hành cao cấp, cảnh báo tài xế giành lại quyền điều khiển (Takeover Request - TOR).
   - Nếu hệ thống thuần camera (Vision-only) không có Radar/LiDAR: Kích hoạt cơ chế dừng xe an toàn tối thiểu (Minimum Risk Maneuver - MRM): giảm dần tốc độ, bật đèn khẩn cấp hazard và tấp vào lề đường.

### 5.3. Danh mục Telemetry cần thu thập trên xe thực tế
Để hiệu chuẩn ngưỡng chính xác ở quy mô hạm đội (fleet deployment), xe cần log các trường sau:
- `frame_timestamp`, `sequence_id`, `camera_id` (trước/sau/hông).
- Thông số ISP phần cứng: `exposure_time` (thời gian phơi sáng), `sensor_gain / ISO`, `shutter_speed`, trạng thái bộ tự động phơi sáng (`auto_exposure_state`).
- Các chỉ số chất lượng ảnh trực tiếp: `laplacian_var`, `highlight_saturation_pct`, `shadow_clip_pct`, `contrast_std`.
- Nhiệt độ cảm biến (`camera_junction_temperature`) và mã chẩn đoán lỗi phần cứng ISP.

### 5.4. Đánh giá Trade-off (Khi nào nên và không nên dùng phương án này)
- **Nên dùng khi:**
  - Hệ thống ADAS chạy trên chip biên nhúng (Edge SoC như Jetson Orin, TDA4VM) đòi hỏi thuật toán kiểm tra sức khỏe siêu nhẹ (< 3 ms), tường minh (interpretable), không tiêu tốn tài nguyên GPU trước khi chạy mô hình AI nặng.
  - Các hệ thống có kiến trúc đa cảm biến (LiDAR + Radar + Camera) cần trọng số động để thực hiện Sensor Fusion an toàn.
- **Không nên dùng khi:**
  - Triển khai nguyên bản mà không có bước thích nghi động (adaptive baseline): Môi trường đêm tối hoàn toàn hoặc đường hầm sẽ làm độ tương phản và biên giảm tự nhiên, nếu dùng ngưỡng cố định sẽ gây báo động giả nhầm lẫn.
  - Sử dụng cho camera quan sát cabin (Driver Monitoring System - DMS): Camera cabin dùng đèn hồng ngoại IR chiếu gần, các đặc tính phân bố sáng và tần số biên hoàn toàn khác biệt so với camera giao thông tầm xa ngoài trời.
