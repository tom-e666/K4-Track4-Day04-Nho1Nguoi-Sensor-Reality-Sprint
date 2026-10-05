# Bản nộp cá nhân — Day 19 · T1 Camera Degradation Health Score

> Số liệu lấy từ bằng chứng chung của nhóm: `results/metrics.csv`, `results/summary.json`, `REPORT.md`.
> **[Đo]** = nhóm tự đo · **[Nguồn]** = paper/repo báo cáo · **[Giả thuyết]** = chưa kiểm chứng.

**Họ tên:** Nguyễn Đức Long  
**Vai trò trong nhóm:** Code: tạo các corruption, tính metric, tích hợp YOLO và BREMOLA (deliverable: `src/run_demo.py`, `tests/`).  
**Repository:** https://github.com/tom-e666/K4-Track4-Day04-Nho1Nguoi-Sensor-Reality-Sprint @ commit `dcf85a55aaaa1aaee5945365c3536df4f7342e99`

## 1. Problem

- **Nền tảng / tính năng / sensor:** Xe ô tô ADAS; tính năng nhận diện vật thể cấp cho các chức năng như AEB/ACC; cảm biến là camera RGB đơn phía trước.
- **Failure case và điều kiện xuất hiện:** Camera bị mờ (blur), chói sáng cục bộ (glare/over-exposure) hoặc nhiễu xung (salt-and-pepper). Detector vẫn có thể xuất box với độ tự tin cao dù ảnh đã hỏng.
- **Claim:** Khi kernel Gaussian blur tăng 5 → 21 px, Laplacian variance giảm và edge retention giảm, nhưng mean confidence của YOLO **không** giảm theo. Vì vậy confidence của detector không thể thay thế một bộ giám sát chất lượng ảnh đầu vào.

## 2. Method

- **Paper/repo đã đọc:**
  - **[Nguồn]** Nam et al., *Vehicles* 2025 (BREMOLA), code https://github.com/woongchan789/BREMOLA @ `7ba26999c265692bb8e44c5a3f2d91c06746830f`. Đây là repo tôi tích hợp vào benchmark.
  - **[Nguồn]** Dong et al. (CVPR 2023) và RoboBEV (TPAMI 2025): chỉ tham khảo ý tưởng "ảnh sạch + một corruption có mức độ".
- **Input → output của phương pháp:**
  - BREMOLA: 1 ảnh RGB → 1 điểm chất lượng no-reference (0–100), dựa trên phổ Fourier và Laplacian.
  - Pipeline tôi viết: ảnh → 1 corruption (tên mang tham số) → metric thô → health score → điểm BREMOLA → YOLOv8n → quy tắc 3 khung liên tiếp.
- **Metric và dataset của nguồn:** **[Nguồn]** BREMOLA đánh giá bằng SROCC trên 45 khung hình × 20 mức blur (900 mẫu). Dong/RoboBEV dùng AP/mAP/NDS trên dataset lớn. Đây là số của họ, không so trực tiếp với số của nhóm.
- **Phần nhóm tái hiện và lý do:**
  - Không chạy lại repo của Dong và RoboBEV vì cần dataset lái xe lớn, mô hình đã huấn luyện và GPU, vượt quá thời gian lab.
  - BREMOLA là script CPU đơn nên tôi chạy **nguyên bản, không sửa**, qua subprocess trên ảnh PNG không nén của từng biến thể.
- **Phần code tôi làm:**
  - `CONDITIONS` + `apply_condition`: 10 điều kiện (1 sạch, 9 lỗi), tên chứa tham số (ví dụ `glare_gain0.70`); tham số đầy đủ (σ blur, tâm/bán kính glare, seed) lưu trong `summary.json`.
  - `raw_metrics`, `add_relative_health`: Laplacian variance, edge density/retention, clipping, entropy, health score.
  - `yolo_detect`, `match_to_reference`: ghép box YOLO với box trên ảnh sạch (cùng lớp, IoU ≥ 0.5) để tìm box mất/box mới.
  - `persistence_policy`, `simulate_policy`: quy tắc 3 khung liên tiếp và mô phỏng chuỗi 23 khung.
  - `run_bremola`, `bremola_meta`: gọi BREMOLA và ghi commit + SHA-256 của script.
  - 12 test trong `tests/test_metrics.py`.

## 3. Benchmark

- **Dữ liệu (thật/tổng hợp), số mẫu, hash/URL:** Một ảnh phố thật `bus.jpg` (810×1080) từ https://ultralytics.com/images/bus.jpg, SHA-256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`. 1 baseline + 9 điều kiện lỗi. Không lưu ảnh gốc trong repo.
- **Baseline và các mức lỗi:**
  - Blur Gaussian: `k=5/11/21` (σ = 1.1 / 2.0 / 3.5 px)
  - Glare: `gain=0.35/0.70/1.05` (đĩa trắng mờ viền, tâm (591, 324) px, bán kính 243 px)
  - Salt-and-pepper: `p=0.5%/1%/2.5%`, seed 19
- **Lệnh chạy:**

```bash
python src/run_demo.py \
  --input assets/bus.jpg \
  --source-url "https://ultralytics.com/images/bus.jpg" \
  --yolo \
  --bremola-script third_party/BREMOLA/bremola.py \
  --output-dir results
pytest -q
```

| Điều kiện | Tham số | Metric (đơn vị) | Giá trị | Bằng chứng |
|---|---|---|---|---|
| Baseline | `clean` | Laplacian var. / Health / YOLO conf. | 3619.17 / 1.000 / 0.656 | `results/metrics.csv` |
| Lỗi A: blur mạnh | `blur_gauss_k21` | Laplacian var. / edge retention / Health / YOLO conf. | 3.40 / 4.1% / 0.444 / 0.675 | `results/metrics.csv`, `results/detector_blur_compare.png` |
| Lỗi B: glare mạnh | `glare_gain1.05` | Pixel cháy sáng / Health / YOLO conf. | 20.00% / 0.636 / 0.549 | `results/metrics.csv`, `results/detector_before_after.png` |
| Lỗi C: nhiễu nhẹ | `saltpepper_p0.005` | Số box / YOLO conf. | 5 / 0.741 | `results/metrics.csv` |

Cả 12 test đều pass và benchmark cho kết quả giống hệt khi chạy lại lần hai (cùng seed).

## 4. Failure case

- **[Đo] Nhóm quan sát được:**
  - Ở `glare_gain1.05`, pixel cháy sáng tăng từ 1.27% lên 20.00%. YOLO **mất `bus 0.87`** và xuất hiện `airplane 0.45` + `truck 0.29`; số box tăng 6 → 7 dù kết quả tệ hơn.
  - Ở `glare_gain0.70` (11.42% pixel cháy sáng) health score vẫn là 0.803 → **OK**, tức bộ giám sát của nhóm bỏ lọt trường hợp này. Lý do: thành phần clipping chỉ có trọng số 0.25 nên riêng glare không thể kéo điểm H xuống dưới 0.75.
  - Ở `blur_gauss_k21`, số box vẫn là 6 nhưng `stop sign 0.26` mất và `dog 0.36` (sai) xuất hiện, nên mean confidence tăng 3.0%.
  - Điểm BREMOLA giảm 24.7% ở k5 rồi đi ngang (29.30 → 29.58 → 28.56), và tăng nhẹ ở glare (+3.8%).
- **[Nguồn] Paper/repo cho biết:** Tác giả BREMOLA giới hạn phương pháp ở blur do lão hoá camera và nêu noise, glare, motion artifact nằm ngoài phạm vi chính. Kết quả đo của nhóm khớp với phạm vi này, không mâu thuẫn với paper.
- **[Giả thuyết] Nguyên nhân / tác động tới tính năng:** Nếu AEB/ACC theo dõi một xe buýt làm xe phía trước, việc box bị đổi lớp đúng lúc đó có thể làm thay đổi quyết định phanh. Nhóm chưa chạy tracker hay planner nên chưa kiểm chứng.
- **Limitation của phép thử:**
  - Chỉ có 1 ảnh phố chụp tay, không phải ảnh từ camera gắn trên xe; lỗi là tổng hợp (glare là đĩa trắng cộng thêm, không có flare).
  - Health score so với ảnh sạch của cùng cảnh, nên **không dùng được nguyên trạng trên xe thật** (xe không có ảnh sạch).
  - Trọng số và ngưỡng 0.65 được chỉnh sau lần chạy đầu rồi cố định; 2 trong 4 mức nhẹ mới bị bỏ lọt.
  - Không có ground truth: "khớp với ảnh sạch" không phải recall. BREMOLA chạy ngoài điều kiện của paper (ảnh bị kéo giãn sang 1920×1080, dùng blur Gaussian thay vì average).

## 5. Engineering decision

- **Cải tiến / fallback đề xuất:**
  1. Không dùng confidence hay số box của detector làm tín hiệu sức khỏe camera.
  2. Thêm quy tắc riêng cho glare (tỉ lệ pixel cháy sáng so với baseline của camera) bên cạnh điểm tổng hợp.
  3. Giữ quy tắc 3 khung liên tiếp: trên chuỗi tổng hợp 23 khung, số cảnh báo nhầm trên khung lỗi thoáng qua giảm 3 → 0, đổi lại báo muộn 2 khung (~67 ms ở 30 fps).
  4. Thay ảnh sạch tham chiếu bằng tham chiếu cuốn chiếu từ các khung khỏe gần nhất.
  - Khi lỗi kéo dài: ghi log, giảm trọng số camera trong fusion, yêu cầu chế độ an toàn hơn, dựa nhiều hơn vào radar/LiDAR nếu có. Repo này **không** cài fusion hay điều khiển.
- **Metric / log kiểm chứng ở vòng sau, tiêu chí đạt:**
  - Tỉ lệ cảnh báo theo từng mức glare trên ≥20 ảnh × 5 mức: mọi mức làm mức khớp ảnh sạch tụt dưới 100% phải được báo ≥90%, tỉ lệ báo nhầm trên ảnh sạch ≤5%.
  - Quy tắc 3 khung trên video thật có gán nhãn đoạn lỗi: 0 cảnh báo trên đoạn lỗi ngắn hơn 3 khung, trễ ≤3 khung khi lỗi kéo dài.
  - Recall thật trên một lát nuScenes-C có nhãn để kiểm tra hoặc bác bỏ proxy hiện tại.
- **Khi nào nên / không nên dùng (trade-off):** Health score hợp để làm công cụ trong phòng lab, so sánh các mức lỗi trên cùng một cảnh. Không nên dùng trên xe thật khi chưa thay phần tham chiếu sạch và chưa hiệu chỉnh ngưỡng trên dữ liệu thật có nhãn. Quy tắc 3 khung giảm cảnh báo nhầm nhưng chậm phát hiện; với chức năng cần phản ứng nhanh thì số khung cần được cân nhắc lại.
