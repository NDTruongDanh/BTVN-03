# ĐÁNH GIÁ HIỆU QUẢ 3 MẪU AGENT

## 1. Phương pháp

- Cùng một `BookingConstraints` (`SGN → DAD`, sáng `2026-10-07`, `≤ 2M`)
  và cùng 5 kịch bản cho cả 3 mẫu (`python evaluate.py`).
- Chỉ tiêu: `status` kết thúc (`SUCCESS` là tốt), số bước tool-call
  (ít hơn là rẻ hơn), và tính trung thực khi thất bại
  (báo `LOOP / STALLED / NEEDS_APPROVAL / FAILED` kèm bàn giao,
  tuyệt đối không bịa mã vé).

| Kịch bản           | Ý nghĩa                                                                |
| ------------------ | ---------------------------------------------------------------------- |
| `standard`         | Đường hạnh phúc, `VN122` thỏa mọi ràng buộc                            |
| `dynamic_sold_out` | `VN122` hết chỗ ngay sau khi search (môi trường biến động)             |
| `over_budget`      | Mọi chuyến sáng đều `> 2M` (nhiệm vụ bất khả thi)                      |
| `timeout_loop`     | `check_seat` luôn timeout (bẫy lặp)                                    |
| `approval_only`    | Chỉ còn vé không hoàn (cần người duyệt; chạy với `auto_approve=False`) |

## 2. Kết quả (`python evaluate.py`)

```
scenario           ReAct                  Plan-then-Exec         Hybrid
--------------------------------------------------------------------------------------
standard           SUCCESS(5)             SUCCESS(5)             SUCCESS(4)
dynamic_sold_out   SUCCESS(5)             FAILED(2)              SUCCESS(4)
over_budget        FAILED(2)              FAILED(5)              FAILED(6)
timeout_loop       LOOP(3)                FAILED(2)              LOOP(5)
approval_only      NEEDS_APPROVAL(4)      FAILED(2)              NEEDS_APPROVAL(3)
```

Tỉ lệ thành công trên 2 kịch bản khả thi (`standard`, `dynamic`):
ReAct 2/2, Hybrid 2/2, Plan 1/2.

## 3. Phân tích

- **`standard` (ai cũng xong):** cả 3 `SUCCESS`. Hybrid ít bước nhất (4)
  vì tin giá listing để chốt `VN122` sớm; ReAct/Plan tốn 5 bước vì
  verify thêm bằng `check_seat` — đắt hơn một chút nhưng chắc chắn hơn
  khi giá listing lỗi thời.
- **`dynamic_sold_out` (thước đo thích nghi):** Plan `FAILED` ở bước 2 —
  lỗi bước đầu làm hỏng toàn bộ phía sau, đúng nhược điểm slide đã nêu.
  ReAct thấy `sold_out` ở V2 và chuyển sang `VN134` (V3–V5) nên `SUCCESS`;
  Hybrid `SUCCESS` nhờ lập lại kế hoạch sau observation đổi.
- **`over_budget` (thước đo trung thực):** cả 3 đều thất bại trung thực,
  không bịa vé `VN999`. ReAct dừng nhanh nhất (2 bước: search xong thấy
  không có ứng viên thỏa ràng buộc thì dừng + bàn giao); Plan chạy hết
  5 bước mù quáng rồi mới rớt kiểm chứng; Hybrid tốn nhất (6) vì cố
  lập lại kế hoạch tìm đường vòng.
- **`timeout_loop` (thước đo harness):** ReAct `LOOP` ở V3 và Hybrid
  `LOOP` ở V5 — bộ `LoopDetector` so `(tool, args)` bắt được việc gọi
  lại `check_seat(VN122)` mà ngân sách vẫn còn, đúng demo 2 trong slide.
  Plan chỉ `FAILED` ở bước 2 (abort khi step lỗi, không phát hiện lặp).
- **`approval_only` (thước đo kiểm quyền):** ReAct/Hybrid dừng
  `NEEDS_APPROVAL` trước khi thực thi, kèm bàn giao
  (“đang ở đâu – định làm gì – vì sao hỏi”). Plan `FAILED` vì kế hoạch
  cứng ghi `VN122` (đã hết chỗ trong kịch bản) nên chưa bao giờ chạm
  tới cổng kiểm quyền — vừa cứng vừa mù rủi ro.

## 4. Kết luận và bảng chọn mẫu

| Mẫu               | Chọn khi                                | Rủi ro chính (kiểm chứng)                                        |
| ----------------- | --------------------------------------- | ---------------------------------------------------------------- |
| ReAct             | Không đoán được số bước, môi trường đổi | Dễ lặp/trôi — đã chặn bằng `LoopDetector` + ràng buộc là dữ liệu |
| Plan-then-Execute | Cần duyệt plan trước, việc ổn định      | Kế hoạch lỗi thời — rớt `dynamic_sold_out`, mù cổng duyệt        |
| Lai               | Việc dài + môi trường biến động         | Debug khó hơn, tốn bước khi nhiệm vụ bất khả thi                 |

Khuyến nghị cho bài đặt vé: dùng **mẫu Lai** (vừa duyệt được todo-list,
vừa thích nghi khi hết chỗ/đổi giá), giữ nguyên toàn bộ harness
(permission trước thực thi, completion/budget kiểm bằng code, bàn giao
30 giây). Plan-then-Execute chỉ nên dùng khi inventory ổn định và bắt
buộc duyệt trước từng bước.
