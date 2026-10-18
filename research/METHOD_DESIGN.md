# Thiết kế phương pháp: phân đoạn cây chết bền vững theo điều kiện thu ảnh

Trạng thái: **bản thiết kế để thảo luận, chưa có kết quả thực nghiệm cho phương pháp.** Mọi số liệu "đã đo" bên dưới
đến từ phân tích dữ liệu hoặc các lần chạy đã thực hiện; mọi hiệu quả kỳ vọng của phương pháp đều là *giả thuyết cần
kiểm chứng*. Nguồn tài liệu ghi ở mục 9, kèm mức xác minh.

---

## 0. Mục tiêu, và "cạnh tranh được" nghĩa là gì

Đối thủ trực tiếp: **TreeMortSeg** (Li et al., Environ. Model. Softw. 2026), baseline mạnh nhất công bố trên TreeFinder.

| Tiêu chí | Ngưỡng cần đạt | Ghi chú |
|---|---|---|
| (a) Split chính thức (theo tile), F1 cây chết | ≥ ~66 (TreeMortSeg 66,29 ± 0,40) | Đây là tiêu chí khó nhất, có khả năng chịu ảnh hưởng của trùng site |
| (b) Split theo site (sạch) | cao hơn TreeMortSeg-nguyên-bản và baseline paper, có ý nghĩa thống kê | Chưa ai báo số này |
| (c) Cross-region W→E / E→W / CO | ≥ 23,7 / 28,8 / 19,6 (số TreeMortSeg báo, 1 lần chạy) | Cần ≥ 3 seed |
| (d) Chi phí | ≤ ~160 GFLOPs @512² (TreeMortSeg 161,7) | Không đánh đổi tính thực tế |

Câu chuyện bài báo nếu (a) hòa nhưng (b), (c) thắng rõ: *"Cùng độ chính xác trong miền, bền hơn khi đổi ảnh và vùng."*

---

## 1. Bằng chứng từ dữ liệu và thí nghiệm (đã đo)

| # | Quan sát | Số liệu | Hệ quả thiết kế |
|---|---|---|---|
| E1 | Vật thể rất nhỏ | trung vị 87 px (~31 m²); 39% thành phần ≤ 64 px; 0,37% pixel là cây chết; 29% site không có cây chết | Cần đường đặc trưng phân giải cao; mất cân bằng cực đoan |
| E2 | Tín hiệu quang phổ một pixel yếu | AUC ngưỡng-chung cây chết vs sống: NDVI 0,72 (thô) → 0,77 (chuẩn hóa tile); các băng khác ≤ 0,70 | Cần ngữ cảnh/kết cấu/hình dạng, không thể dựa vào màu |
| E3 | Dịch chuyển nằm giữa các **site** | phương sai của trung bình tile: giữa bang 18–25%, **giữa site cùng bang 44–52%**, trong site 23–36% (NIR: giữa bang 47%) | Bất biến theo điều kiện thu ảnh; đơn vị nhóm = site |
| E4 | Chuẩn hóa giúp, tương phản cục bộ không thêm | AUC NDVI: tile‑z 0,771, site‑z 0,758, local‑contrast‑31 0,748 | Dùng chuẩn hóa theo tile; **không** thêm nhánh tương phản cục bộ |
| E5 | Kiến trúc gần như không phân biệt được trên split sạch | split theo site, 1 seed: U‑Net 0,5M 0,303; U‑Net‑R34 0,299; DeepLabV3+ 0,302; SegFormer‑B2 0,322, CI bootstrap ±0,13 | Độ lợi của backbone lớn trên split theo tile có thể một phần là *ghi nhớ site* |
| E6 | Lỗi nằm ở xếp hạng, không ở ngưỡng | F1 ở ngưỡng oracle chỉ hơn ngưỡng chọn trên validation 0,00–0,04; AP 0,05–0,17 | Không cần module hiệu chỉnh ngưỡng phức tạp |
| E7 | Dịch chuyển miền đảo chiều precision/recall | W→E: P≈0,4, R≈0,1; CO→khác: R 44%, P 4% (U‑Net) | Mô hình không bền về hiệu chỉnh ở miền mới |
| E8 | Split chính thức trùng site | 857/860 site test có tile trong train/val | Báo kết quả trên cả hai split |
| E9 | Nhãn phụ trong mã TreeMortSeg công bố sinh từ vùng **no‑data** | trên 4 tile thật: cạnh/khoảng cách phụ trùng cây chết = 0 pixel | Cơ chế multi‑task trong bài chưa được chứng minh; cần cài đặt đúng và đo lại |
| E10 | Thêm thành phần multi‑task chỉ +1,76 điểm trong bài, backbone chiếm phần lớn | Table 3: nền 64,53 → đủ 66,29; DeepLabV3+ 59,39 | Phải kiểm soát backbone khi so sánh |

---

## 2. Bản đồ tài liệu: mỗi hướng làm gì và để lại khoảng trống nào

| Hướng | Đại diện | Cái đã giải quyết | Khoảng trống với TreeFinder |
|---|---|---|---|
| Kiến trúc cho vật thể nhỏ/biên | TreeMortSeg, TreeMort‑3T‑UNet (3 đầu: mask, tâm Gaussian, SDT‑biên; hậu xử lý watershed) | Biên, tách cây | Đánh giá trên split theo tile; không giải quyết dịch chuyển miền; (TreeMortSeg) nhãn phụ nghi sai |
| Mô hình nền lớn đa độ phân giải | DT‑V1 / Möhring et al. 2025 (SegFormer‑B5 + U‑Net, Focal‑Tversky, cân bằng bốn chiều: biome/độ phân giải/có cây chết/nguồn) | Khái quát qua biome và độ phân giải cm; lấy mẫu cân bằng | Precision rất thấp trên TreeFinder (37,8%); nhãn crowd‑sourced |
| Thích nghi miền không giám sát | ADA‑Net (CUT + chú ý + tương phản tần số, dịch ảnh Mỹ→Phần Lan) | Giảm dịch chuyển phóng xạ bằng dịch ảnh | Cần ảnh miền đích khi huấn luyện, một cặp miền, không có seed/độ lệch chuẩn |
| Khái quát miền (DG) cho viễn thám | frequency‑based style mix (TGRS 2023); style mapping + prototypical contrast; disentangle content–style (FLAIR) | Randomize/loại bỏ style | Chưa áp dụng cho cây chết; chưa có benchmark TreeFinder |
| DG tổng quát | SWAD (trung bình trọng số dày, tìm cực tiểu phẳng); GroupDRO | Bền vững bằng tối ưu | Chưa dùng *site* làm nhóm trong phân đoạn cây chết |
| Tiền huấn luyện đúng miền | SatlasPretrain (có NAIP), SatMAE, DOFA | Biểu diễn viễn thám | DOFA kém trong paper TreeFinder (F1 29,2) — nhưng chạy RGB‑only đối với các transformer khác; chưa rõ |
| Hàm mất mát cho vật thể nhỏ | Dice/Tversky/Focal‑Tversky; boundary loss (Kervadec) | Mất cân bằng | Boundary loss nhạy với thang đo |

Khoảng trống chung: **chưa có phương pháp nào được thiết kế quanh dịch chuyển theo ảnh (site) cho cây chết và được đánh giá trên split theo site cùng cross‑region nhiều seed.**

---

## 3. Phương pháp đề xuất: AIR‑Seg (Acquisition‑Invariant Robust Segmentation)

Ba tầng, mỗi tầng ứng với một nhóm bằng chứng. Tất cả cắm được vào ConvNeXt‑UPerNet (cùng họ với TreeMortSeg để kiểm soát backbone), và các tầng 1, 3 cắm được vào mô hình bất kỳ của paper gốc.

```
                    ┌── Tầng 1: đầu vào bất biến ───────────────────────┐
 RGB-NIR (+NDVI) ──►│ chuẩn hóa tile (robust z) ⊕ aug phóng xạ theo band │
                    └───────────────────────────┬───────────────────────┘
                                                ▼
                    ┌── Tầng 2: mạng đa tỉ lệ + nhánh cấu trúc ─────────┐
                    │ ConvNeXt-S/B → PPM + FPN (UPerNet, 1/4)            │
                    │ + đường phân giải cao 1/2 (skip từ stem)           │
                    │ đầu: mask │ SDT-trong-tán │ biên │ (tâm)            │
                    └───────────────────────────┬───────────────────────┘
                                                ▼
                    ┌── Tầng 3: huấn luyện nhận biết site ──────────────┐
                    │ lấy mẫu cân bằng site/có‑cây‑chết • phạt phương   │
                    │ sai giữa site (V-REx/GroupDRO) • SWA dày • TTA     │
                    └────────────────────────────────────────────────────┘
```

### Tầng 1 — Bất biến đầu vào (E2, E3, E4)
- **Chuẩn hóa tile bền vững**: với mỗi tile, trừ trung vị và chia cho độ lệch tuyệt đối trung vị trên pixel hợp lệ (mỗi kênh). Bằng chứng: tăng AUC NDVI +0,05 (E4).
- **Aug phóng xạ theo band**: gain, bias, gamma ngẫu nhiên theo từng kênh trước khi tính NDVI; mô phỏng khác biệt ngày chụp/cảm biến (E3). Đã có tín hiệu cross‑region (1 seed).
- **Lý do không thêm tương phản cục bộ**: E4 cho thấy không cải thiện thêm.

### Tầng 2 — Đa tỉ lệ và nhánh cấu trúc *đúng* (E1, E9, E10)
- Backbone ConvNeXt‑S (khớp TreeMortSeg) và biến thể ConvNeXt‑B; neck UPerNet.
- **Đường phân giải cao**: thêm skip ở 1/2 vào giải mã cuối, vì vật thể trung vị ~9×9 px chỉ chiếm ~2×2 ô ở 1/4. (Giả thuyết; kiểm bằng ablation, và biến thể thay thế: nâng đầu vào ×2.)
- **Nhánh phụ với nhãn đúng** (`label == 1`, loại no‑data):
  - SDT trong tán, chuẩn hóa theo từng tán (tâm = 1), hồi quy L1 chỉ trong tán;
  - biên 1‑pixel (Dice trên mục tiêu 0/1 chuẩn);
  - tùy chọn: bản đồ tâm Gaussian (như 3T‑UNet) để dùng cho hậu xử lý tách cây.
- Mask loss giữ **đúng loss của benchmark** (BCE + Dice) để so sánh công bằng; Focal‑Tversky là một ablation riêng.
- Vì E9, bắt buộc có ba dòng đối chứng: *không nhánh phụ*, *nhánh phụ như mã công bố (no‑data)*, *nhánh phụ đúng*.

### Tầng 3 — Huấn luyện nhận biết site (E1, E3, E5, E7)
- **Lấy mẫu cân bằng**: mỗi batch lấy đều theo site và theo "có/không cây chết" (cùng ý với cân bằng bốn chiều của DT‑V1). 10% site giữ 72% pixel chết; không cân bằng thì mô hình bị vài site chi phối.
- **Phạt phương sai giữa các site** (V‑REx) hoặc GroupDRO theo site: tổn thất trung bình mỗi site được tính trong batch; cộng λ·Var. Site là nhóm tự nhiên (không cần gán nhãn thêm).
- **SWA dày kiểu SWAD**: trung bình trọng số mỗi bước trong khoảng chọn theo validation loss; kết hợp được với mọi mô hình, thường cho +1–2 điểm DG trên DomainBed (số của tác giả SWAD, +1,6 trung bình).
- **TTA lật/quay** lúc suy luận (rẻ, nhất quán).

---

## 4. Vì sao kỳ vọng nó cạnh tranh được (lập luận và giới hạn)

1. **Trên split sạch/cross‑domain** (E3, E5): TreeMortSeg tối ưu cấu trúc nhưng không có cơ chế nào chống dịch chuyển theo ảnh; tầng 1 và 3 nhắm đúng nguồn dịch chuyển lớn nhất đã đo. Đây là nơi có cơ hội thắng rõ.
2. **Trên split theo tile** (a): nơi khó nhất. Nếu độ lợi của ConvNeXt+UPerNet phần lớn là ghi nhớ site (E5, giả thuyết), phương pháp bất biến có thể *hòa hoặc thấp hơn một chút*. Để cạnh tranh ở (a) cần: backbone ngang TreeMortSeg (khớp), nhánh cấu trúc đúng, SWA, TTA, và có thể ConvNeXt‑B. Chưa thể đảm bảo vượt 66,29.
3. **Về nhánh phụ**: nếu (đúng nhãn) cho độ lợi thật, đó là một kết quả mới. Nếu bằng 0, đó cũng là kết quả (E9) và ta vẫn có tầng 1 và 3.
4. **Mức tăng đáng tin**: trên cross‑region kỳ vọng +2–4 điểm F1 (cùng cỡ chênh giữa TreeMortSeg và baseline); chưa chắc chắn. DomainBed cho thấy ERM rất khó thắng, nên thắng thật phải đo bằng nhiều seed.

Điều **không** nên hứa: thắng ở mọi kịch bản, hoặc thắng lớn ở split theo tile.

---

## 5. Các thí nghiệm cổng (rẻ, quyết định trước khi xây hết)

| Cổng | Câu hỏi | Chạy gì | Quyết định |
|---|---|---|---|
| G1 | Lợi thế backbone có phải ghi nhớ site? | ConvNeXt‑S‑UPerNet *chỉ mask* trên split theo tile và theo site, so với DeepLabV3 và U‑Net, 3 seed | Nếu lợi thế biến mất trên site → định vị bài là "DG", đừng chạy đua trên (a) |
| G2 | Nhánh phụ đúng có giúp thật? | không phụ / phụ no‑data (như repo) / phụ đúng, 3 seed | Nếu đúng ≈ không phụ → bỏ nhánh phụ, giảm chi phí |
| G3 | Phóng xạ + tile‑norm có bền không? | bật/tắt trên 8 kịch bản chính thức, 3 seed | Giữ nếu trung bình qua kịch bản tăng có kiểm định |
| G4 | Cân bằng site + phương sai + SWA | thêm từng cái, split site + cross‑region | Giữ từng thành phần theo ablation |
| G5 | Phân giải cao | nâng đầu vào ×2 hoặc skip 1/2 | Giữ nếu bù được chi phí |

Mỗi cổng chạy trong chính `main.py` (thêm cờ, không đổi mặc định), cùng recipe paper.

---

## 6. Giao thức đánh giá

- **Split**: chính thức (8 kịch bản) + site‑disjoint (seed 2025/2024/2023) + đối chứng tile/site cùng kích thước.
- **Seed**: 3 (2025, 2024, 2023); báo mean ± sd; kiểm định theo cặp (bootstrap theo site).
- **Chỉ số**: đúng `exps/evaluate.py` (ngưỡng 0,5) + AP, F1 mức đối tượng, FP trên mỗi ha hợp lệ.
- **Baseline**: U‑Net, DeepLabV3, SegFormer, Mask2Former (mã gốc, đã tái lập); TreeMortSeg (mã công bố, hai bản: nguyên bản, sửa nhãn).
- **Chọn tham số**: trên validation của miền nguồn và split theo site; **không** chọn trên test chính thức.

---

## 7. Ma trận ablation (bắt buộc đưa vào bài)

| Thành phần | Bật/tắt | Metric chính |
|---|---|---|
| Chuẩn hóa tile | ✓/✗ | F1 cross‑region trung bình |
| Aug phóng xạ | ✓/✗ | F1 cross‑region |
| Nhánh SDT / biên (đúng nhãn) | ✗ / no‑data / đúng | F1 site, F1 tile |
| Đường 1/2 hoặc ×2 | ✓/✗ | F1, recall vật thể nhỏ |
| Cân bằng site | ✓/✗ | F1 site |
| Phạt phương sai giữa site | ✓/✗ | độ lệch giữa site, F1 |
| SWA | ✓/✗ | F1 cross‑region |
| TTA | ✓/✗ | F1 |
| Backbone | ConvNeXt‑S/B | F1, GFLOPs |

---

## 8. Rủi ro và phương án thay thế

| Rủi ro | Dấu hiệu | Phương án |
|---|---|---|
| Không vượt TreeMortSeg trên split theo tile | G1 cho thấy lợi thế do ghi nhớ | Định vị bài là DG + đánh giá sạch; báo cả hai split |
| Nhánh phụ đúng không giúp | G2 | Bỏ nhánh; bài còn tầng 1 và 3 |
| Cải thiện nhỏ hơn nhiễu seed | kiểm định không có ý nghĩa | Tăng seed, gộp nhiều kịch bản, nêu thẳng giới hạn |
| Backbone ConvNeXt‑B quá nặng | GFLOPs > ngưỡng | Giữ ConvNeXt‑S |
| Phạt giữa site làm giảm hiệu năng | F1 giảm trên split sạch | Giảm λ, hoặc chỉ dùng cân bằng site |

---

## 9. Tài liệu và mức xác minh

| Tài liệu | Dùng cho | Mức xác minh |
|---|---|---|
| TreeFinder (NeurIPS 2025) | dataset, baseline, split | đọc toàn văn |
| TreeMortSeg (Li et al., 2026) + mã `RONGHUA-LI/TreeMortSeg` | đối thủ, kiểm tra mã | đọc toàn văn + mã, chạy kiểm tra nhãn phụ |
| TreeMort‑3T‑UNet, arXiv 2503.21438 | 3 đầu, watershed | đọc HTML qua agent; một số bảng từ tóm tắt tự động |
| ADA‑Net, arXiv 2504.04271 | thích nghi miền bằng dịch ảnh | đọc HTML qua agent |
| Möhring et al. 2025 (DT‑V1), epic.awi.de/id/eprint/60469 | SegFormer‑B5 + U‑Net, Focal‑Tversky, cân bằng bốn chiều | **chỉ tóm tắt/kết quả tìm kiếm** |
| SWAD, arXiv 2102.08604 (NeurIPS 2021) | trung bình trọng số dày | **chỉ tóm tắt/kết quả tìm kiếm** |
| SatlasPretrain, arXiv 2211.15660 | tiền huấn luyện có NAIP | **chỉ tóm tắt** (chưa kiểm tra có dùng được cho 4 kênh) |
| Kervadec et al., boundary loss (MIDL 2019) | số hạng khoảng cách | **từ trí nhớ, chưa kiểm tra lại** |
| Sagawa et al., GroupDRO (ICLR 2020), Krueger et al., V‑REx (ICML 2021) | huấn luyện bền theo nhóm | **từ trí nhớ, chưa kiểm tra lại** |
| DG viễn thám: frequency‑based style mix (TGRS 2023, doi 10.1109/TGRS.2023.3344670), style mapping + prototypical contrast, disentangle content–style (FLAIR) | tham khảo hướng DG | **chỉ tóm tắt** |
| DINOv2 cho phân đoạn UAV (arXiv 2507.04915) | hướng thay thế backbone (patch 14 bất lợi cho vật thể nhỏ) | chỉ tóm tắt |

Những mục "chưa kiểm tra" phải được đọc lại trước khi trích dẫn trong bài.
