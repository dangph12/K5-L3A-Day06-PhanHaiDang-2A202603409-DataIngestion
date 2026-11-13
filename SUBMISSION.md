# Submission — bài cá nhân

## Repo bài nộp

Fork repo đề bài hoặc clone rồi đẩy lên repo cá nhân. Đặt tên gợi ý `K5-L3A-DAY06-MSSV-DataIngestion` hoặc `K5-L3B-DAY06-MSSV-DataIngestion`; theo hướng dẫn lớp nếu có quy ước khác. Điền [STUDENT.md](STUDENT.md) và nộp link repo theo kênh giảng viên/coach chỉ định. Thời hạn do giảng viên/coach thông báo.

## Cấu trúc tối thiểu

```text
STUDENT.md
README.md                         # cách setup, chạy, đường dẫn dữ liệu
src/ hoặc starter/src/             # code pipeline do học viên viết
data/manifest.*                   # nguồn, trang, quyền, cách lấy
benchmark/                        # câu hỏi, split, gold evidence
artifacts/experiment_log.csv      # cấu hình và giả thuyết A/B/C
results/                          # thống kê, trace, nhãn audit, log
report/individual/<MSSV>_<HoTen>.md
```

Mẫu báo cáo: [report/individual/MSSV_HoTen.md](report/individual/MSSV_HoTen.md). `src/` có thể đặt ở root hoặc `starter/src/`; benchmark và dữ liệu thật do học viên tạo. Nếu không được phép nộp dữ liệu gốc hoặc trace chứa nội dung bị hạn chế, cung cấp hướng dẫn truy cập/tái tạo và minh chứng được phép chia sẻ; báo cho người chấm biết cách xác minh.

## Checklist

- [x] Dữ liệu ≥30 trang nội dung hợp lệ, manifest ghi nguồn, quyền, số trang và quy tắc tính.
- [x] ≥20 câu theo [BENCHMARK.md](BENCHMARK.md), có evidence đã kiểm chứng và tách tối thiểu 15 dev / 5 test trước thử nghiệm.
- [x] Có embedding model thật và vector store hỗ trợ lưu bền vững, vector search, metadata filter; ghi phiên bản và cấu hình.
- [x] Có ba cấu hình, một so sánh thay đúng một yếu tố; cùng model, corpus và benchmark.
- [x] Chạy đủ và lưu trace top-3, chỉ số dev/test; ghi chi phí, model và thời gian.
- [x] Có audit thủ công và phân tích cả 5 khía cạnh theo GUIDE; ghi giới hạn.
- [x] Có lệnh chạy, code, manifest, experiment log và báo cáo cá nhân.
- [x] Không có secret, dữ liệu riêng tư hoặc dữ liệu không được phép chia sẻ.
- [x] Repo mở được cho người chấm và đã nộp link theo hướng dẫn lớp.
