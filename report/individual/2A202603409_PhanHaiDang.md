# Báo cáo cá nhân — Day 6

- **Tác giả:** Phan Hải Đăng (2A202603409, L3A)
- **Lần chạy:** `results/20261113T103132732631Z/`

## Dữ liệu và pipeline

- Chủ đề: câu lệnh và biểu thức Python trong _Introduction to Python Programming_, OpenStax / Rice University. PDF ghi giấy phép CC BY-NC-SA 4.0; cần giữ thông tin ghi công. Nguồn, ngày truy cập, cách lấy và quy tắc chọn trang nằm trong `data/manifest.json`.
- Dữ liệu gồm 60 trang vật lý của PDF, từ 17 đến 76, thuộc hai chương Statements và Expressions; loại phần đầu trang 1–16. Đây là số trang vật lý, không phải số in trên sách. Pipeline từ chối trang trích xuất rỗng nhưng không tự đánh giá tính hữu ích của từng trang.
- Trích xuất bằng chế độ layout của `pypdf`, chuẩn hóa NFC và khoảng trắng. Mỗi chunk giữ ID tài liệu, URL nguồn, trang vật lý, chương và vị trí ký tự. Trường `chapter` lấy từ ranh giới chương đã xác minh trong manifest; hai giá trị hỗ trợ lọc bằng phép so sánh bằng nhưng chưa xác định được mục nhỏ cụ thể.
- Embedding: Gemini API `gemini-embedding-001`, 768 chiều; `RETRIEVAL_DOCUMENT` cho chunk, `RETRIEVAL_QUERY` cho câu hỏi; `google-genai` 1.75.0. Lưu trữ bằng PostgreSQL 16.16, pgvector 0.8.7, tìm kiếm cosine chính xác. Chạy `python -m src.run` trong `starter/` sau khi cài database và cấu hình `GEMINI_API_KEY` theo `starter/README.md`.
- Tổng thời gian ghi nhận: 35.79 giây. **Chưa đo chi phí API**, không được xem là 0. `pipeline.log` lưu diễn biến chạy; `run_info.json` lưu phiên bản, hash, thời gian và bản chụp đầu vào.

## Cấu hình và benchmark

| Cấu hình | Chia chunk                        | Bộ lọc                                 | Số chunk | Tổng ký tự | Nạp (giây) | Nạp lại (giây) |
| -------- | --------------------------------- | -------------------------------------- | -------: | ---------: | ---------: | -------------: |
| A        | Cửa sổ 500 ký tự, overlap 50      | Không                                  |      258 |    114,467 |       7.60 |           0.66 |
| B        | Nhóm đoạn văn, tối đa 500 ký tự   | Không                                  |      265 |     99,758 |       9.21 |           0.58 |
| C        | Dùng lại chunk và embedding của B | So sánh bằng chapter khi được cung cấp |      265 |     99,758 |       1.00 |           0.52 |

Ba cấu hình dùng cùng dữ liệu, model và `benchmark/questions.json`: 20 câu gồm 15 dev và 5 test. Có ba câu test cung cấp bộ lọc chương; dev không có câu lọc. Bằng chứng và trang vật lý được đối chiếu với PDF trước truy xuất. A–B thay cách chia chunk; B–C chỉ thay bộ lọc lúc truy vấn. Nhật ký: `artifacts/experiment_log.csv`.

Quy tắc chọn đã ghi trước khi xem test: ưu tiên Evidence Hit@3 dev, sau đó Hit@1 dev, rồi tổng ký tự được lập chỉ mục ít hơn. Quy tắc chọn **A** với Hit@3 dev 14/15, so với 13/15 của B và C. Kết quả test không làm thay đổi lựa chọn đã ghi.

## Kết quả

| Cấu hình | Tập  | Hit@1 | Evidence Hit@3 |
| -------- | ---- | ----: | -------------: |
| A        | Dev  |  9/15 |          14/15 |
| A        | Test |   3/5 |            4/5 |
| B        | Dev  | 10/15 |          13/15 |
| B        | Test |   4/5 |            5/5 |
| C        | Dev  | 10/15 |          13/15 |
| C        | Test |   4/5 |            5/5 |

| Cấu hình | Coherence trung bình /2 (10 vị trí; số nhãn 0/1/2) | Metadata đủ và đúng | Tuân thủ bộ lọc test | Grounding 0/1/2 (5 câu test); trung bình /2 |
| -------- | -------------------------------------------------: | ------------------- | -------------------- | ------------------------------------------- |
| A        |                                   0.40 (7 / 2 / 1) | 15/15    | 3/3, quan sát được   | 1 / 1 / 3; 1.40                             |
| B        |                                   1.30 (0 / 7 / 3) | 15/15    | 3/3, quan sát được   | 0 / 1 / 4; 1.80                             |
| C        |                                   1.30 (0 / 7 / 3) | 15/15    | 3/3, có áp dụng lọc  | 0 / 1 / 4; 1.80                             |

## Phân tích và kết luận

- **A–B:** B tăng coherence từ 0.40 lên 1.30 và Hit@3 test từ 4/5 lên 5/5, nhưng Hit@3 dev giảm từ 14/15 xuống 13/15. B có thêm 7 chunk, giảm 14,709 ký tự và nạp lâu hơn (9.21 so với 7.60 giây). A vẫn là cấu hình được chọn theo dev.
- **B–C:** C dùng lại chunk và vector của B. Các câu dev không lọc cho kết quả giống nhau; ba câu test có lọc giữ nguyên Hit@1, Hit@3 và điểm grounding. C thực thi giới hạn chương; mức tuân thủ 3/3 của B là kết quả quan sát. C nạp nhanh hơn do tái sử dụng embedding, không phải do lọc làm giảm chi phí tạo embedding.
- **Lỗi q06 (A, trang PDF 24):** Hai kết quả đầu thuộc trang gold nhưng không chunk nào chứa đủ trích dẫn về toán tử gán. Vì vậy Evidence Hit@3 thất bại dù truy xuất đúng trang.
- **Lỗi q01 (B/C, trang PDF 17):** Ba kết quả đầu thuộc trang 48, 47 và 48 thay vì trang gold 17; thiếu đoạn bằng chứng về trình tự câu lệnh.

Nếu làm lại, cần lấy chi phí từ nhà cung cấp, xác nhận độc lập nhãn metadata và thiết kế câu dev có bộ lọc trước khi kết luận rộng hơn về lợi ích của C.
