# BÁO CÁO BTVN#3 — Agent đặt vé máy bay bằng LangChain

## 1. Mục tiêu

Xây dựng agent đặt vé máy bay khứ hồi một chặng `SGN → DAD`, sáng ngày
`2026-10-07`, ngân sách `≤ 2.000.000 VND`, theo đúng kiến trúc buổi 03:

```
agent = goal + tools + loop + termination
```

Chỉ bước “model đề xuất tool” là của model; 4 bước còn lại
(dựng ngữ cảnh, gọi tool, ghi kết quả, xét điều kiện dừng) là **harness**
do code của nhóm thực hiện.

## 2. Goal

Trạng thái cần đạt (không phải danh sách bước):

> Đặt được một vé `SGN → DAD` bay sáng `2026-10-07`, giá `≤ 2.000.000 VND`,
> trạng thái `confirmed` và đã thanh toán.

## 3. Tools mockup (`flight_tools.py`)

| Tool | Vai trò | Observation chuẩn hóa |
|---|---|---|
| `search_flights(origin, destination, date)` | Liệt kê chuyến | `{status: ok, flights: [...]}` |
| `check_seat(flight_id)` | Giá + ghế trống | `{status: ok, flight: {...}}` |
| `book_seat(flight_id)` | Giữ chỗ | `{status: held, code, booking}` |
| `pay(code)` | Thanh toán | `{status: paid, code, booking}` |
| `get_booking(code)` | Đọc lại để kiểm chứng | `{status: ok, booking}` |

Mọi tool trả về `dict` có trường `status` tường minh:
`ok / held / paid / invalid_param / not_found / sold_out / error`,
kèm `hint` hướng phục hồi (sửa đúng 4 failure mode trong slide:
sai định dạng ngày → gợi `YYYY-MM-DD`; kết quả rỗng → `{status: ok, flights: []}`
thay vì `{}` mơ hồ). File này cũng expose các hàm `raw_*` để harness
dùng trực tiếp và 5 tool `@tool` để cắm vào `create_agent`.

Dữ liệu mẫu (ngày `2026-10-07`): `VN122` 08:10 / 1.850.000 (đáp án đúng),
`VJ604` 09:30 / 2.080.000 (quá ngân sách), `QH118` 15:40 / 1.640.000
(sai giờ), `VN134` 10:20 / 1.950.000 (không hoàn → cần duyệt).

## 4. Các lớp harness (`harness.py`)

| Lớp | Ánh xạ slide | Cài đặt |
|---|---|---|
| Ràng buộc là dữ liệu | `CONSTRAINTS = Constraints(date=..., depart_before=..., max_price=...)` | `BookingConstraints` + `satisfies()` / `progress_score()` |
| Tiêu chí hoàn thành kiểm bằng code | `get_booking(c).status == "confirmed" and paid and price <= 2M and date/time` | `check_completion()` — sensor computational, 0 token; cộng `cross_check_price()` chống bịa giá |
| Kiểm quyền (chạy **trước** thực thi) | Vé không hoàn / vượt hạn mức → chờ duyệt | `PermissionChecker.check()` — checklist #0 |
| Phát hiện lặp / bế tắc | So `(tool, args)`, đo đại lượng tiến triển | `LoopDetector` (y nguyên code slide) |
| Ngân sách (kiểm **cuối cùng**) | Trần vòng / thời gian | `BudgetTracker` — kiểm sau cùng để không che mất chẩn đoán |
| Bàn giao | Trạng thái + đã thử gì + 1 câu hỏi 30 giây | `build_handoff()` + `TraceLogger` + `AgentResult` |

Thứ tự checklist sau mỗi observation: hoàn thành → lặp → bế tắc → ngân sách.

## 5. Ba mẫu thiết kế (mỗi mẫu một file `.py`)

- **`agent_react.py` — ReAct:** lặp Suy luận → Hành động → Quan sát.
  Mỗi vòng chọn đúng 1 tool từ observation mới nhất; dừng khi model thôi
  gọi tool **và** `check_completion()` xác nhận bằng code.
  LangChain: `create_agent(..., middleware=[ModelCallLimitMiddleware(run_limit=12)])`.
- **`agent_plan_execute.py` — Plan-then-Execute:** một lần gọi model sinh
  trọn kế hoạch 5 bước (`make_plan`), duyệt plan, rồi thực thi tuần tự
  **không lập lại kế hoạch** — bước lỗi là abort toàn bộ.
  LangChain: thêm `HumanInTheLoopMiddleware(interrupt_on={book_seat, pay})`.
- **`agent_hybrid.py` — Lai (ReAct + Plan):** lập kế hoạch → thực thi `k=2`
  bước → nếu observation đổi đáng kể (hết chỗ, đổi giá) thì lập lại kế hoạch.
  LangChain: `TodoListMiddleware` + `ModelCallLimitMiddleware`.

Ba file dùng chung `flight_tools.py` và `harness.py` (không copy-paste).
Vì môi trường chấm bài chưa chắc có API key, mỗi file có chính sách
quyết định dựa trên luật (rule-based, đóng vai model) để chạy offline
tái hiện được; khi có `MODEL_NAME` (ví dụ `openai:gpt-4o-mini`) thì hàm
`build_*_langchain_agent()` cắm model thật vào cùng tools + harness.

## 6. Hướng dẫn chạy

```bash
pip install -r requirements.txt
python agent_react.py          # ReAct, kịch bản chuẩn
python agent_plan_execute.py   # Plan-then-Execute
python agent_hybrid.py          # Lai
python evaluate.py              # so sánh 3 mẫu trên 5 kịch bản
```

Dùng model thật (tùy chọn — copy `.env.example` thành `.env` rồi điền model):

```bash
copy .env.example .env   # Windows; điền MODEL_NAME trong .env
python -c "from agent_react import build_react_langchain_agent; build_react_langchain_agent()"
```

## 7. File nộp

- `flight_tools.py`, `harness.py` — dùng chung.
- `agent_react.py`, `agent_plan_execute.py`, `agent_hybrid.py` — 3 mẫu.
- `evaluate.py` — chạy đánh giá; `requirements.txt`.
- `BAO_CAO.md` (file này), `DANH_GIA_HIEU_QUA.md` — đánh giá 3 mẫu.
