# Developer TODO

- [ ] **days_pattern in `parser.py`**
    - Is there a better way to determine the pattern without listing all possible patterns?

- [ ] **preprocess_image in `parser.py`**
    - Implement proper deskewing process

- [ ] **Clean subject name in `parser.py`**
    - Subject = after code and before units
    - Does not check the validity of string. Example: "_— Artificial Intelligence"

- [x] **class_pattern in `parser.py`**
    - Only considers "BSCS". Consider broadening this by a more general pattern like `r"\\b[A-Z]{2,5}\\b"`.