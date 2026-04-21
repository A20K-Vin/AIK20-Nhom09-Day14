# Reflection Report - Nguyen Thuy Linh

## 1) Vai tro va pham vi dong gop

Trong bai lab Day 14, em dam nhan vai tro **Data/SDG Lead**. Em phu trach xay dung va chot bo test du lieu dau vao cho toan bo he thong danh gia, bao gom:

- Tao va can bang `data/golden_set.jsonl` thanh 50 test case.
- Bao dam phan bo do kho theo muc tieu: **15 easy / 15 medium / 20 hard**.
- Bo sung hard cases theo guide: adversarial, ambiguous, out-of-context, conflicting information, multi-turn.
- Chuan hoa metadata (difficulty, type, hard_case_tag, section, ground_truth_ids) de cac thanh vien retrieval/judge/QA co the phan tich va debug co he thong.

Muc tieu cua em la tao mot bo du lieu "on dinh va co the kiem chung" de benchmark V1 vs V2 cong bang, han che tinh trang ket qua dao dong vi data thay doi.

## 2) Engineering Contribution

Em co cac dong gop ky thuat cu the, co the kiem chung qua file va output:

- Cap nhat truc tiep `data/golden_set.jsonl` tu bo placeholder thanh bo 50 case dua tren tai lieu `data/docs/data.md`.
- Dieu chinh theo `data/HARD_CASES_GUIDE.md` de tang do kho thuc chien cho bo test.
- Gan nhan hard-case cho cac truong hop kho de support failure clustering.
- Cap nhat `data/synthetic_gen.py` theo huong tao du lieu dung schema va validate split difficulty.
- Kiem tra lai so luong va split bang script (50 cases, dung 15/15/20).

Dong gop nay tao nen "data contract" thuc te cho team:

- Nguoi 2 (Retrieval): dung metadata va ground-truth de tinh Hit Rate, MRR.
- Nguoi 3 (Judge): dung hard-case taxonomy de phan tich agreement va conflict.
- Nguoi 5 (QA/Release): dung bo data freeze de regression V1 vs V2.

## 3) Technical Depth

Qua vai tro Data/SDG Lead, em hieu sau hon cac khai niem danh gia:

- **MRR phu thuoc truc tiep vao ground truth**: neu mapping ground-truth sai, MRR cao/thap deu mat y nghia.
- **Position bias**: retrieval co the lay duoc chunk dung nhung xep hang thap, dan den quality danh gia generation bi anh huong.
- **Judge agreement (vd. Cohen's Kappa)** can bo data du "sach" va case kho da dang de phan biet ro bat dong that su va nhiu do du lieu.
- **Trade-off chi phi - chat luong**: hard cases giup phat hien loi that su nhung tang token/cost; vi vay can strategy chay 2 tang (quick set + full set).

Em da dua cac loai case kho vao bo test de do nang luc agent trong cac tinh huong de hallucination va sai logic, thay vi chi test fact-check don gian.

## 4) Problem Solving

Trong qua trinh lam, em gap mot so van de va cach xu ly nhu sau:

1. **Van de: bo test ban dau qua it va thieu do kho**
   - Xu ly: mo rong len 50 case, phan bo do kho ro rang, bo sung hard-case taxonomy.

2. **Van de: de bi lech ket qua khi team tu sua data**
   - Xu ly: chot bo golden set theo ban freeze de benchmark nhat quan.

3. **Van de: khong co nhan metadata de phan tich loi sau benchmark**
   - Xu ly: bo sung `metadata.type`, `difficulty`, `hard_case_tag`, `section` de failure analysis co cau truc.

4. **Van de: rui ro danh gia retrieval sai neu map ID khong chuan**
   - Xu ly: uu tien thiet ke case co context ro, giup doi retrieval doi chieu va debug mapping.

## 5) Ket qua, bai hoc va huong cai tien

### Ket qua

- Bo test 50 case da duoc tao va can bang theo muc tieu do kho.
- Team co bo du lieu nhat quan de chay benchmark tong.
- Cac truong hop kho giup nang chat luong failure analysis thay vi chi nhin diem trung binh.

### Bai hoc

- Data quality quyet dinh do tin cay cua moi metric phia sau.
- Freeze dataset la buoc quan trong de dam bao regression cong bang.
- Metadata dung ngay tu dau giup giam rat nhieu thoi gian debug cuoi ky.

### Huong cai tien

- Viet them script kiem tra ton tai cua `ground_truth_ids` voi index that.
- Them `data_contract.md` va `dataset_changelog.md` de version hoa du lieu ro rang.
- Tach mot subset nho de smoke-test nhanh, sau do chay full set cho bao cao chinh thuc.
