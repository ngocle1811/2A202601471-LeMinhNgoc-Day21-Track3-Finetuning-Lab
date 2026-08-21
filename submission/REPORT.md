# Lab 21 — Evaluation Report

**Họ tên**: Lê Minh Ngọc  
**MSSV**: 2A202601471  
**Ngày**: 21/08/2026  
**Tier**: `T4`  
**Base model**: `unsloth/Qwen3.5-4B`  
**GPU thực tế**: NVIDIA T4, 16 GB danh nghĩa (peak VRAM đo được của run `correct`: 12,01 GB)

> Mọi con số trong báo cáo được chép từ các artefact trong `results/`; tập đánh giá đầy đủ gồm 50 mẫu target và 15 mẫu regression, không dùng `EVAL_LIMIT`.

---

## 1. Setup

| Hạng mục | Cấu hình |
|---|---|
| Dataset | Corpus mặc định: 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25, split cố định với seed 42 |
| `max_length` | 1024 theo cấu hình tier T4; p95 đo được là 98 token, giá trị gợi ý là 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epoch / 30 optimizer step |
| Precision | fp16 |
| Batch hiệu dụng | 16, nhỏ hơn ngưỡng 32 |

`max_length=1024` lớn hơn mức 256 được gợi ý từ p95. Tôi giữ cấu hình chuẩn của tier T4 để các run trong lab dùng chung một cấu hình và chắc chắn không cắt mất câu trả lời; đổi lại, đây chưa phải lựa chọn tối ưu về hiệu suất cho corpus ngắn này. Nếu tối ưu lại pipeline triển khai, tôi sẽ thử cap 256 và xác nhận lại rằng điểm target, format và regression không đổi.

**Template có giữ khối `<think>` không?** Có. `template_check.json` ghi nhận cả thẻ mở và nội dung reasoning đều còn trong chuỗi render, với verdict `reasoning preserved — safe to train on traces`; vì vậy không cần thay chat template. Corpus được phát hành chỉ chứa câu trả lời JSON trần, nên việc giữ `<think>` không đồng nghĩa mô hình được huấn luyện trên reasoning trace trong lần chạy này.

---

## 2. Mask proof (NB1)

| Hạng mục | Kết quả |
|---|---:|
| `supervised_fraction` | 0,3936 (37/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |

Đoạn được tính loss là câu trả lời JSON và token kết thúc của assistant. Dữ liệu gốc lưu JSON trên một dòng; dưới đây chỉ xuống dòng để dễ đọc:

```text
{"intent": "doi_tra", "urgency": "trung_binh",
 "product": "balo laptop",
 "sentiment": "trung_tinh"}
<|im_end|>
```

Mask này chứng minh loss không bao phủ prompt của system hoặc câu hỏi người dùng. Nếu dùng `everything`, model sẽ bị thưởng cả khi học chép lại đầu vào; `assistant-only` giữ đúng tín hiệu giám sát là câu trả lời cần sinh.

---

## 3. Ba baseline (NB2 — đo trước khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0,000 | 0,7578 | 0,000 | 3240,9 |
| (b) base + optimized prompt | 0,765 | 0,7578 | 1,000 | 1006,7 |
| (c) LoRA fine-tune | 0,975 | 0,7000 | 1,000 | 1445,2 |

**Baseline (b) có thật sự mạnh hơn (a) không?** Có. Chỉ riêng prompt có schema, miền giá trị và ví dụ đã đưa target từ 0,000 lên 0,765 và format từ 0,000 lên 1,000; đồng thời latency đo được còn giảm từ 3240,9 ms xuống 1006,7 ms. Vì vậy (b), chứ không phải prompt ngây thơ (a), là đối thủ công bằng của bản fine-tune.

Tôi không sửa `OPTIMIZED_PROMPT`: SHA được đóng băng là `719e74d3b6232053` và gatekeeper xác nhận prompt không đổi. Việc đóng băng prompt và tập eval trước khi train tránh điều chỉnh mốc so sánh sau khi đã nhìn thấy kết quả của adapter.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí | r | Trainable | LR | Train loss | Target | Thời gian (s) | VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear (12 module) | 16 | 32.464.896 | 1e-4 | 0,6290 | 0,975 | 962,5 | 12,01 |
| `attn_only` | q,v (2 module) | 283, matched | 32.456.704 | 1e-4 | 0,5376 | 0,970 | 786,9 | 12,02 |
| `wrong_lr` | text-linear (12 module) | 16 | 32.464.896 | 1e-5 | 1,5704 | 0,000 | 922,4 | 12,01 |
| `qlora` | text-linear (12 module), 4-bit | 16 | 32.464.896 | 1e-4 | 0,7058 | 0,940 | 982,4 | 7,09 |

Cả bốn run dùng 30 step. `attn_only` chỉ lệch 8.192 tham số so với `correct`, tương đương khoảng 0,0252%, nên phép so sánh vị trí adapter nằm trong ngưỡng công bằng 5%.

### 4.1 — Rank so với vị trí gắn adapter

Trên tập target, `attn_only` đạt 0,970 và thua nhẹ `correct` 0,975, tức chênh 0,005 dù rank đã được nâng từ 16 lên 283 để khớp ngân sách tham số. Thứ tự theo train loss lại ngược: `attn_only` có loss 0,5376, thấp hơn 0,6290 của `correct`, nhưng không có target tốt hơn. Kết quả này cho thấy tăng rank để dồn cùng số tham số vào q,v không thay thế hoàn toàn việc đặt adapter trên toàn bộ linear layer; vị trí là đòn bẩy có ý nghĩa, còn loss train thấp không đủ để tuyên bố thắng. Tuy nhiên, hai điểm target rất sát nhau cũng cho thấy tác vụ JSON triage hẹp có thể được giải quyết gần như hoàn chỉnh chỉ với attention, nên bằng chứng ủng hộ `text-linear` ở đây là có thật nhưng không lớn.

### 4.2 — Learning rate sai

`wrong_lr` chỉ hạ learning rate từ 1e-4 xuống 1e-5, trong khi placement, rank, số tham số và 30 step được giữ nguyên. Artefact chỉ lưu loss cuối chứ không lưu toàn bộ đường cong, nên tôi không suy diễn hình dạng từng step; số đo chắc chắn là loss cuối 1,5704, cao khoảng 2,5 lần `correct` 0,6290. Quan trọng hơn, target và format đều bằng 0,000, cho thấy với cùng ngân sách step, LR ở thang full fine-tuning làm adapter học quá chậm. Nếu chỉ nhìn thấy loss có giảm mà không biết LR và không chấm target, tôi có thể kết luận sai rằng run vẫn đang học và chỉ cần tin vào chỉ số huấn luyện; phép đánh giá ngoài tập train cho thấy cấu hình này chưa học được hành vi đầu ra cần thiết.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ 12,01 GB xuống 7,09 GB, tiết kiệm 4,92 GB, tương đương khoảng 41,0%. Cái giá là target giảm từ 0,975 xuống 0,940, train loss tăng từ 0,6290 lên 0,7058 và latency đánh giá tăng từ 1445,2 ms lên 1780,2 ms (khoảng 23,2%); format vẫn giữ 1,000. Trên T4 nơi LoRA 16-bit vẫn vừa bộ nhớ, số đo này ủng hộ khuyến nghị không chọn QLoRA làm cấu hình mặc định cho Qwen3.5: tiết kiệm bộ nhớ là thật nhưng chất lượng target kém hơn. Nếu thiết bị chỉ có khoảng 8 GB VRAM thì QLoRA vẫn là một phương án thực dụng, nhưng đó là đánh đổi có đo lường, không phải một cải tiến miễn phí.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0,210` · `regression Δ = -0,0578` · `valid_trace_rate = 0,00`

Fine-tune cải thiện mạnh đúng nhiệm vụ: target tăng từ 0,765 của baseline prompt tốt lên 0,975, format giữ ở 1,000. Tuy nhiên, regression giảm từ 0,7578 xuống 0,7000, tức giảm 0,0578 và vượt ngưỡng chịu đựng 0,020 của cổng an toàn. Vì vậy verdict `FAILED` không có nghĩa quá trình train hỏng; nó có nghĩa model đã chuyên môn hóa thành công nhưng đánh đổi quá nhiều năng lực phổ thông. Với một model vẫn phải xử lý câu hỏi chung, tôi không nên triển khai adapter này ở trạng thái hiện tại. Hướng sửa hợp lý là thêm khoảng 1–5% replay data phổ thông vào tập SFT, giữ nguyên eval đã đóng băng, rồi đo lại cả target và regression. `valid_trace_rate=0,00` không đủ để kết luận reasoning bị collapse, bởi corpus và đầu ra mong muốn đều là JSON trần, không yêu cầu reasoning trace.

---

## 6. Định tính — có cả ca đúng và ca sai

`qualitative.json` lưu prediction của fine-tune nhưng không lưu prediction từng mẫu của baseline (b). Vì vậy tôi không bịa lại output của (b); “đúng/thua” trong bảng dưới được xác định trực tiếp bằng nhãn gold và `ft_score`. Điểm tổng của (b) trên toàn tập là 0,765.

| # | Ticket rút gọn | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---:|---|---|---|---|---|
| 1 | Trả chuột không dây, “Gấp”, shop hỗ trợ tốt | `doi_tra / cao / chuột không dây / tich_cuc` | Không lưu prediction/mẫu | Đúng 4/4 (`1,00`) | ✅ FT đúng hoàn toàn |
| 2 | Hoàn tiền ốp lưng, “Sớm nhé”, bực mình | `hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc` | Không lưu prediction/mẫu | Đúng 4/4 (`1,00`) | ✅ FT đúng hoàn toàn |
| 3 | Chưa thấy tiền bình giữ nhiệt, “Khi nào tiện” | `hoan_tien / thap / bình giữ nhiệt / tich_cuc` | Không lưu prediction/mẫu | Dự đoán urgency `trung_binh` (`0,75`) | ❌ FT sai urgency |
| 4 | Áo khoác gió bị lỗi, “Khi nào tiện” | `san_pham_loi / thap / áo khoác gió / tich_cuc` | Không lưu prediction/mẫu | Dự đoán urgency `trung_binh` (`0,75`) | ❌ FT sai urgency |
| 5 | “Trả lại tiền” máy xay, “Không vội” | `hoan_tien / thap / máy xay sinh tố / trung_tinh` | Không lưu prediction/mẫu | Dự đoán intent `doi_tra` (`0,75`) | ❌ FT nhầm intent |

Mẫu chung rõ nhất ở các ca sai là cụm mức độ nhẹ như “Khi nào tiện” bị đẩy từ `thap` lên `trung_binh`; hai trong ba ví dụ thua mắc đúng lỗi này. Ca còn lại cho thấy cụm “trả lại tiền” mơ hồ giữa hành động đổi/trả hàng (`doi_tra`) và hoàn tiền (`hoan_tien`). Model đã học rất chắc schema và product, nhưng ranh giới ngữ nghĩa giữa các nhãn gần nhau vẫn là nơi cần thêm dữ liệu khó hoặc quy tắc gán nhãn rõ hơn.

---

## 7. Kết luận và điều tôi học được

Tôi chưa nên deploy bản fine-tune này cho một trợ lý đa năng, dù target 0,975 và format 1,000 rất hấp dẫn. Nguyên nhân là cổng hồi quy đo được mức giảm 0,0578, gần ba lần ngưỡng cho phép 0,020; deploy ngay sẽ đổi độ chính xác triage lấy sự suy giảm ở các yêu cầu phổ thông. Trong bối cảnh chỉ cần một bộ phân loại chuyên dụng, adapter vẫn là ứng viên đáng xem xét, nhưng baseline (b) cũng là lựa chọn mạnh: không cần train, giữ regression 0,7578, target 0,765 và latency chỉ 1006,7 ms. Đòn bẩy lớn nhất trong thí nghiệm này không phải rank. Bằng chứng là `attn_only` dùng rank 283 và gần như cùng số tham số vẫn không vượt `correct`, trong khi chỉ giảm LR mười lần làm target rơi về 0. Mask là điều kiện đúng từ gốc; nếu mask sai thì mọi so sánh sau đó mất ý nghĩa. Chất lượng và độ phủ dữ liệu mới là bước tiếp theo cần ưu tiên, đặc biệt là replay data để chống quên thảm họa và các ví dụ biên phân biệt `thap/trung_binh` cùng `doi_tra/hoan_tien`. Sau khi bổ sung dữ liệu, tôi sẽ giữ nguyên baseline và eval đã đóng băng để biết cải thiện có thật hay chỉ là thay thước đo.

**Ba điều tôi học được:**

1. Một prompt có schema và ví dụ có thể thay đổi kết quả rất lớn: target tăng từ 0,000 lên 0,765 mà chưa cập nhật bất kỳ trọng số nào.
2. Loss train không phải bảng xếp hạng năng lực: `attn_only` có loss thấp nhất nhưng target vẫn kém `correct`, còn `wrong_lr` cho thấy cùng kiến trúc nhưng sai thang LR có thể làm tác vụ thất bại hoàn toàn.
3. Fine-tuning tốt trên tác vụ đích chưa đủ để deploy; target tăng 0,210 nhưng regression giảm 0,0578 nên cần một cổng đánh giá đa chiều thay vì chỉ báo cáo accuracy đẹp nhất.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1%, 3% và 5% replay data phổ thông vào cùng corpus, giữ `assistant-only`, placement `text-linear`, LR 1e-4 và cùng ngân sách step; sau đó chọn tỷ lệ nhỏ nhất giúp regression trở lại trong ngưỡng 0,020 mà không làm mất phần lớn mức tăng target. Tôi cũng sẽ sửa bước đánh giá để lưu prediction từng mẫu của baseline (b), nhờ đó có thể thực hiện so sánh định tính head-to-head thay vì chỉ dựa vào điểm tổng.

---

## Phụ lục — thưởng đã làm

- [x] B1 NB6 merge: điểm trước và sau merge đều 0,975 trên 50 mẫu, delta 0,000; notebook cũng thực hiện hot-swap các adapter đã tạo trong NB4 trên cùng base.
- [ ] B2 dataset miền riêng.
- [ ] B3 reasoning-trace collapse với hai `MASK_MODE`.
- [ ] B4 quét rank có kiểm soát.
- [ ] B5 HuggingFace Hub.
