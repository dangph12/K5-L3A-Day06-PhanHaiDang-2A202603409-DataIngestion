# PostgreSQL + pgvector

Starter này hỗ trợ thí nghiệm truy xuất tài liệu với ba cấu hình: A chia chunk theo độ dài cố định, B theo đoạn văn và C dùng chunk của B kèm metadata filter.

## Cài đặt

Yêu cầu Python 3.11 trở lên và Docker. Chạy trong thư mục `starter/`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền `DATABASE_URL` và `GEMINI_API_KEY` trong `.env`. Gemini tạo embedding 768 chiều; cần kết nối Internet và có thể phát sinh phí. Không chia sẻ API key.

Khởi động và kiểm tra PostgreSQL:

```powershell
docker compose -p day6_student_starter up -d --wait
python scripts/check_setup.py
```

## Dữ liệu và benchmark

Cập nhật `../data/manifest.json` với nguồn, quyền sử dụng, PDF, quy tắc chọn trang và metadata đã xác minh. Chọn ít nhất 30 trang nội dung hữu ích; loại bìa, mục lục, trang trống và trang chỉ có tài liệu tham khảo.

Chuẩn bị `../benchmark/questions.json` với ít nhất 20 câu hỏi, chia tối thiểu 15 câu dev và 5 câu test. Mỗi câu cần đáp án, trang nguồn và bằng chứng đã kiểm tra; có ít nhất 3 câu metadata filter, trong đó một câu thuộc test.

## Chạy

```powershell
python -m src.run
python -m unittest tests.test_pipeline -v
```

Mỗi lần chạy tạo thư mục kết quả riêng trong `../results/`. Bổ sung số liệu vào `../artifacts/experiment_log.csv` và hoàn tất đánh giá thủ công, báo cáo theo [mẫu báo cáo](../REPORT_TEMPLATE.md).

Pipeline thay toàn bộ dữ liệu của các cấu hình trong bảng `day6_chunks` mỗi lần nạp. Không nạp một phần tài liệu; sao lưu dữ liệu cần giữ trước khi chạy lại. Dừng Docker và giữ dữ liệu bằng lệnh:

```powershell
docker compose -p day6_student_starter stop
```

Xem [GUIDE.md](../GUIDE.md) để biết quy trình và cách đánh giá, [BENCHMARK.md](../BENCHMARK.md) về benchmark, và [SUBMISSION.md](../SUBMISSION.md) về yêu cầu nộp bài.
