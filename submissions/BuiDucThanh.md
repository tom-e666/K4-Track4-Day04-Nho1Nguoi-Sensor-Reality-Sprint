# Bản nộp cá nhân — Day 19 · T1 Camera Degradation Health Score

**Họ tên:** Bùi Đức Thành  
**Vai trò trong nhóm:** Đọc và tổng hợp tài liệu về camera IQA, camera soiling và corruption robustness; tìm hiểu BREMOLA và cách chạy repo gốc để đối chiếu với benchmark của nhóm.  
**Repository:** https://github.com/tom-e666/K4-Track4-Day04-Nho1Nguoi-Sensor-Reality-Sprint @ commit `8f85299447a6c2925b174ed35ffcf945b7a1d6e5`

## 1. Problem

- **Nền tảng / tính năng / sensor:** ADAS trên ô tô, tập trung vào camera RGB phía trước dùng cho object detection và các chức năng downstream như AEB/ACC.

- **Failure case và điều kiện xuất hiện:** Camera có thể bị giảm chất lượng do optical blur, glare/over-exposure hoặc sensor/ISP noise. Vấn đề là detector vẫn có thể tạo bounding box với confidence cao ngay cả khi chất lượng ảnh đã giảm mạnh.

- **Claim:** **[Đo]** Khi Gaussian blur tăng từ ảnh clean đến `blur_gauss_k21`, Laplacian variance giảm từ `3619.17` xuống `3.40` px-intensity² và edge retention giảm từ `100%` xuống `4.1%`. Tuy nhiên mean confidence của YOLOv8n lại tăng từ `0.656` lên `0.675`. Vì vậy confidence của detector không thể được dùng thay thế cho camera-health metric.

## 2. Method

- **Paper/repo đã đọc:**
  - **[Nguồn]** Dong et al., CVPR 2023, *Benchmarking Robustness of 3D Object Detection to Common Corruptions*. Paper tạo KITTI-C, nuScenes-C và Waymo-C với nhiều loại controlled corruption để đánh giá robustness của 3D detector.
  - **[Nguồn]** Xie et al., TPAMI 2025, *RoboBEV*. Paper đánh giá các BEV model dưới nhiều camera corruption/failure khác nhau và cho thấy performance trên clean data chưa đủ để thể hiện robustness.
  - **[Nguồn]** Nam et al., Vehicles 2025, *BREMOLA*. Phương pháp nhận một RGB frame và trả về no-reference blur-quality score dựa trên Fourier spectrum và Laplacian. Nhóm chạy code upstream tại commit `7ba26999c265692bb8e44c5a3f2d91c06746830f`.
  - **[Nguồn]** Yang et al., Sensors 2026, nghiên cứu camera-lens soiling severity và temporal stabilization.

- **Input → output của phương pháp:**

```text
RGB camera frame
→ controlled degradation
→ image-health metrics
→ composite health score
→ BREMOLA no-reference score
→ pretrained YOLOv8n
→ detector proxy
→ persistence policy
```

- **Metric và dataset của nguồn:**  
  **[Nguồn]** Dong et al. và RoboBEV dùng các autonomous-driving dataset lớn và các metric như AP/mAP, NDS, RCE, mCE và mRR. BREMOLA sử dụng 45 driving frames với 20 mức blur và đánh giá bằng SROCC. Yang et al. dùng các metric như Spearman correlation và temporal Jitter (MAD).

- **Phần nhóm tái hiện và lý do:**  
  **[Đo]** Do không đủ thời gian và tài nguyên để chạy lại KITTI-C, nuScenes-C hoặc RoboBEV, nhóm chỉ tái hiện ý tưởng controlled corruption: dùng cùng một ảnh baseline và thay đổi từng corruption với tham số cụ thể. BREMOLA đủ nhẹ nên nhóm chạy trực tiếp code upstream không chỉnh sửa trên tất cả các biến thể ảnh.

## 3. Benchmark

- **Dữ liệu:** **[Đo]** Một ảnh street scene thật `bus.jpg` từ Ultralytics, kích thước `810×1080`. SHA-256: `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63`.

- **Số mẫu:** 1 baseline và 9 degraded conditions thuộc 3 nhóm:
  - Gaussian blur: `k=5`, `k=11`, `k=21`
  - glare: `gain=0.35`, `0.70`, `1.05`
  - salt-and-pepper noise: `p=0.005`, `0.010`, `0.025`

- **Lệnh chạy:**

```bash
python src/run_demo.py \
  --input assets/bus.jpg \
  --source-url https://ultralytics.com/images/bus.jpg \
  --yolo \
  --bremola-script third_party/BREMOLA/bremola.py \
  --output-dir results
```

| Điều kiện | Tham số | Metric | Giá trị | Bằng chứng |
|---|---|---|---:|---|
| Baseline | `clean` | Laplacian / Health / YOLO conf. | `3619.17 / 1.000 / 0.656` | `results/metrics.csv` |
| Blur mạnh | `blur_gauss_k21`, σ=`3.5 px` | Laplacian / Edge retention / Health | `3.40 / 4.1% / 0.444` | `results/metrics.csv`, `results/detector_blur_compare.png` |
| Glare mạnh | `glare_gain1.05` | Highlight saturation / Health / Agreement | `20.00% / 0.636 / 83%` | `results/metrics.csv`, `results/detector_before_after.png` |
| Noise nhẹ | `saltpepper_p0.005` | YOLO detections / mean conf. | `5 / 0.741` | `results/metrics.csv` |

**[Đo]** Một kết quả đáng chú ý là tại `blur_gauss_k21`, ảnh bị blur rất mạnh nhưng mean confidence của YOLO tăng khoảng 3%. Ở `saltpepper_p0.005`, YOLO mất một box confidence thấp nên mean confidence còn tăng từ `0.656` lên `0.741`, tức khoảng 13.1%.

## 4. Failure case

- **[Đo] Nhóm quan sát được:**  
  Ở `glare_gain1.05`, highlight saturation tăng từ `1.27%` lên `20.00%`. YOLO mất detection `bus 0.87` nhưng lại xuất hiện `airplane 0.45` và `truck 0.29`; detection count tăng từ 6 lên 7 dù output thực tế đã xấu hơn.

  Với `blur_gauss_k21`, detection count vẫn giữ 6 nhưng `stop sign 0.26` biến mất và xuất hiện `dog 0.36`. Điều này cho thấy chỉ nhìn số lượng detection hoặc mean confidence có thể gây hiểu nhầm.

  Với salt-and-pepper noise, Laplacian variance tăng thay vì giảm: tại `p=0.025`, giá trị tăng từ `3619.17` lên `13968.22`. Vì vậy một rule đơn giản kiểu “Laplacian càng cao thì ảnh càng tốt” không hoạt động với noise.

- **[Nguồn] Paper/repo cho biết:**  
  Dong et al. và RoboBEV cho thấy perception model có thể giảm robustness đáng kể dưới image corruption. BREMOLA tập trung chủ yếu vào blur và không được thiết kế để xử lý glare/noise. Trong benchmark của nhóm, BREMOLA giảm từ `38.91` xuống khoảng `29` khi có blur nhưng lại tăng dưới glare và noise, phù hợp với phạm vi của phương pháp gốc.

- **[Giả thuyết] Nguyên nhân / tác động tới tính năng:**  
  Detector confidence chỉ được tính trên những bounding box mà model quyết định giữ lại. Khi một box confidence thấp biến mất hoặc bị thay bằng một prediction sai nhưng confidence cao hơn, mean confidence có thể tăng dù chất lượng perception thực tế giảm. Với ADAS, nếu lead vehicle bị mất hoặc đổi class ở thời điểm quan trọng thì có thể ảnh hưởng tới tracking hoặc quyết định AEB/ACC, nhưng nhóm chưa chạy tracker hay planner nên chưa kiểm chứng tác động này.

- **Limitation của phép thử:**  
  Benchmark chỉ dùng một ảnh street scene và các corruption tổng hợp. Không có ground-truth annotation nên `agreement with clean pseudo-labels` không phải recall hay mAP. Health score hiện tại còn reference-based, tức cần ảnh clean của cùng scene nên chưa thể deploy trực tiếp trên xe. BREMOLA cũng được chạy khác điều kiện paper gốc vì nhóm dùng Gaussian blur trong khi paper sử dụng average-filter blur.

## 5. Engineering decision

- **Cải tiến / fallback đề xuất:**
  1. Không sử dụng YOLO confidence hoặc detection count làm camera-health signal.
  2. Kết hợp nhiều image-health signal thay vì chỉ Laplacian hoặc một metric duy nhất.
  3. Thêm glare-specific rule dựa trên highlight saturation để bổ sung cho composite health score.
  4. Dùng persistence: `health_score < 0.65` hoặc glare flag trong 3 frame liên tiếp mới chuyển sang `CAMERA_DEGRADED`.
  5. Khi deploy thực tế, thay clean-reference bằng rolling reference từ các frame gần đây được đánh giá healthy hoặc sử dụng thêm no-reference metrics.
  6. Nếu hệ thống có radar/LiDAR, khi camera degraded có thể giảm trọng số camera và dựa nhiều hơn vào sensor độc lập.

- **Metric / log để kiểm chứng vòng sau:**  
  **[Đo]** Trong synthetic sequence 23 frame hiện tại, single-frame rule tạo 3 alarm trên transient corruption, trong khi 3-frame persistence giảm xuống 0 alarm và phát hiện persistent blur sau 2 frame.

  Vòng sau nên thử trên ít nhất nhiều ảnh/video khác nhau và ghi lại false-positive rate, detection latency, highlight saturation, health components và detector recall trên dữ liệu có ground truth.

  Tiêu chí đề xuất trong report gồm: clean false-flag rate ≤5%, các mức glare làm detector agreement giảm phải được flag ≥90%, và persistence không alarm với corruption ngắn hơn 3 frame.

- **Trade-off:**  
  Persistence giảm false alarm nhưng tạo thêm latency. Composite reference-based score dễ kiểm tra trong lab nhưng không phù hợp trực tiếp với môi trường xe đang chạy. BREMOLA có ưu điểm no-reference nhưng chủ yếu nhạy với blur, vì vậy vẫn cần metric riêng cho glare và noise.