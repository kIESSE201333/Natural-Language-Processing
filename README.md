# Natural Language Processing – RLHF (PPO) & Machine Translation (Transformers/GPT)

Repository lưu **báo cáo + notebook** cho đồ án NLP, gồm 2 phần:

1) **Câu 1 – RLHF / PPO**: minh hoạ PPO (CartPole) và PPO cho LLMs theo hướng RLHF (TRL + Transformers).  
2) **Câu 2 – Machine Translation (EN↔VI)**: so sánh mô hình **tự huấn luyện (no-pretrain)** và **pretrained**:
   - GPT (no-pretrain) + SentencePiece tokenizer
   - GPT-2 pretrained (fine-tune)
   - Transformer (no-pretrain)
   - MarianMT pretrained (Helsinki-NLP)

---

## Mục lục
- [Tổng quan](#tổng-quan)
- [Cấu trúc repo](#cấu-trúc-repo)
- [Tech Stack](#tech-stack)
- [Datasets](#datasets)
- [Hướng dẫn chạy](#hướng-dẫn-chạy)
- [Chi tiết từng notebook](#chi-tiết-từng-notebook)
  - [Câu 1](#câu-1--rlhf--ppo)
  - [Câu 2](#câu-2--machine-translation-envi)
- [Ghi chú tái lập kết quả](#ghi-chú-tái-lập-kết-quả)
- [Nhóm thực hiện](#nhóm-thực-hiện)

---

## Tổng quan

Repo này chứa:
- **Report**: PDF + DOCX
- **Notebooks**:
  - PPO trên môi trường CartPole (giúp hiểu PPO “cơ bản”)
  - PPO cho LLMs (RLHF-style) với thư viện TRL/Transformers
  - MT (EN-VI): huấn luyện từ đầu và fine-tune pretrained

> Lưu ý: Một số notebook đang dùng đường dẫn kiểu **Kaggle** (`/kaggle/input/...`). Khi chạy local, bạn cần thay đổi đường dẫn dữ liệu.

---

## Cấu trúc repo

Khuyến nghị tổ chức lại khi public GitHub:

```
.
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ report/
│  ├─ 52200206_52200214_52200216.pdf
│  └─ 52200206_52200214_52200216.docx
└─ notebooks/
   ├─ q1_ppo_cartpole.ipynb
   ├─ q1_ppo_llms_rlhf.ipynb
   ├─ q2_gpt_no_pretrain.ipynb
   ├─ q2_gpt2_pretrained.ipynb
   ├─ q2_transformer_no_pretrain.ipynb
   └─ q2_marianmt_pretrained.ipynb
```

---

## Tech Stack

- **Python 3.10+** (khuyến nghị)
- **PyTorch** (core training)
- **Transformers / Datasets** (Hugging Face)
- **TRL** (PPO / RLHF utilities)
- **SentencePiece** (tokenizer cho GPT no-pretrain)
- **Evaluate / SacreBLEU / ROUGE** (đánh giá MT)
- **Gym** (CartPole PPO demo)
- Jupyter / tqdm / pandas / numpy / scikit-learn

---

## Datasets

Các notebook MT (Câu 2) dùng dữ liệu song ngữ **English–Vietnamese** (các file `train.en.txt`, `train.vi.txt` hoặc IWSLT'15 en-vi),
nhưng **dataset không nằm trong repo** (để tránh nặng).

### Option A — Chạy trên Kaggle (giữ nguyên notebook)
Notebook đã set sẵn đường dẫn kiểu:
- `/kaggle/input/data-nlp/train.en.txt`
- `/kaggle/input/data-nlp/train.vi.txt`
- hoặc IWSLT'15 en-vi trong `/kaggle/input/iwslt15-englishvietnamese/...`

=> Chỉ cần add dataset đúng tên vào Kaggle Dataset/Notebook environment.

### Option B — Chạy local (khuyến nghị khi public GitHub)
1) Tải dữ liệu về (từ Kaggle / nguồn bạn dùng trong lớp)
2) Đặt vào thư mục ví dụ:
   - `data/mt/train.en.txt`
   - `data/mt/train.vi.txt`
3) Sửa trong notebook:
   - `en_path = "data/mt/train.en.txt"`
   - `vi_path = "data/mt/train.vi.txt"`

> Nếu bạn muốn repo “chuẩn GitHub”: thêm file `data/README.md` ghi nguồn dataset + cách tải (không commit dữ liệu thô).

---

## Hướng dẫn chạy

### 1) Tạo môi trường
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Cài dependencies
```bash
pip install -r requirements.txt
```

### 3) Mở notebook
```bash
jupyter lab
```

---

## Chi tiết từng notebook

### Câu 1 – RLHF / PPO

**1) PPO CartPole**  
Notebook gợi ý: `q1_ppo_cartpole.ipynb` (từ file `ppo-for-cartpole-v1.ipynb`)

- Mục tiêu: hiểu PPO qua bài toán RL kinh điển (CartPole-v1)
- Thành phần: policy/value network, advantage estimation, clipping objective, training loop
- Output: reward curve / performance theo episode

**2) PPO cho LLMs (RLHF-style)**  
Notebook gợi ý: `q1_ppo_llms_rlhf.ipynb` (từ file `ppo_llms_rlhf.ipynb`)

- Dùng **Transformers + TRL** để minh hoạ PPO fine-tuning cho mô hình causal LM
- Có setup dependencies (torch/transformers/trl/accelerate/peft…)
- Có bước chuẩn hoá dữ liệu song ngữ (lọc câu dài, tách train/val/test) theo notebook

> Chạy local có thể cần GPU. Nếu CPU-only, hãy giảm batch size / sequence length để test logic.

---

### Câu 2 – Machine Translation (EN↔VI)

Mục tiêu: so sánh hướng **train từ đầu** vs **fine-tune pretrained**.

**1) GPT no-pretrain** (`gpt-nopre.ipynb`)
- Tạo tokenizer bằng **SentencePiece**
- Encode dữ liệu → xây dataset → train mô hình kiểu GPT “nhỏ”
- Đánh giá bằng BLEU/ROUGE (tuỳ cell)

**2) GPT-2 pretrained** (`GPT2.ipynb`)
- Dùng tokenizer pretrained của GPT-2 + thêm special tokens (ví dụ: `[EN]`, `[VI]`)
- Fine-tune mô hình GPT-2 trên tập dữ liệu EN-VI
- Đánh giá (một số cell dùng `evaluate`)

**3) Transformer no-pretrain** (`Transformer_NoPretrain.ipynb`)
- Implement Transformer seq2seq (encoder-decoder) từ đầu
- Preprocess & filter sentence length
- Train + evaluate (BLEU/ROUGE tuỳ phần)

**4) MarianMT pretrained** (`helsinki-marianmtmodel.ipynb`)
- Dùng model pretrained của Helsinki-NLP (MarianMT)
- Fine-tune và đánh giá **SacreBLEU**
- Dataset ví dụ: IWSLT'15 en-vi (theo notebook)

---

## Ghi chú tái lập kết quả

- Training từ đầu (GPT/Transformer) có thể **tốn thời gian**; khuyến nghị GPU.
- Một số notebook cài package trực tiếp trong notebook (`pip install ...`). Nếu bạn chạy local, có thể bỏ các cell đó sau khi đã dùng `requirements.txt`.
- Kết quả MT sẽ dao động theo seed, hyperparameters, và preprocessing. Nếu cần tái lập: set seed cho `random/numpy/torch`.

---

## Nhóm thực hiện

- 52200206
- 52200214 – Trần Hồ Hoàng Vũ
- 52200216
