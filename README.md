# SchedBuddy-ML

**Project Overview**
- **Purpose:**: Small OCR + image-processing toolkit that extracts schedule data from images and demonstrates text recognition workflows.
- **Files:**: `parser.py`, `tesseract.py`, `ocr-ml-system.py`, `cv2-demo.py`, `extracted_schedule.json`, `ocr_raw.txt`, and an `images/` folder.

**Prerequisites**
- **Python:**: Python 3.10+ recommended. Ensure `python` is on your `PATH`.
- **Tesseract OCR:**: Install Tesseract OCR (system binary) for your OS.
  - Windows installer: https://github.com/tesseract-ocr/tesseract
  - After install, ensure the Tesseract installation folder (e.g. `C:\Program Files\Tesseract-OCR`) is in your `PATH` or set the `TESSERACT_CMD` environment variable.

**Quick Setup (Windows PowerShell)**
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Upgrade pip and install Python dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. Verify Tesseract is available:

```powershell
tesseract --version
```

If `tesseract` is not found, add its install directory to your `PATH` or set the `TESSERACT_CMD` environment variable used by `pytesseract`.

Example (set `TESSERACT_CMD` for the current user):

```powershell
[Environment]::SetEnvironmentVariable("TESSERACT_CMD","C:\\Program Files\\Tesseract-OCR\\tesseract.exe","User")
# Restart your shell after this change
```

If PowerShell blocks activation scripts, allow local scripted activation for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Run Examples**
- **Run the parser on an image:**

```powershell
python parser.py images/sample-cor.png
```

- **Run demo scripts:**

```powershell
python cv2-demo.py
python ocr-ml-system.py
python tesseract.py
```

**Repository Notes**
- **Dependencies:**: See `requirements.txt` (OpenCV, pytesseract, Pillow, numpy).
- **Sample data:**: `extracted_schedule.json` and `ocr_raw.txt` contain sample output and intermediate OCR text.
- **Images:**: Put test images in the `images/` directory. Use descriptive file names for easier testing.

**Troubleshooting**
- **pytesseract.TesseractNotFoundError:**: Ensure `tesseract` is installed and either on `PATH` or `TESSERACT_CMD` is set.
- **OpenCV import errors:**: Make sure `opencv-python` is installed inside the activated venv.
- **Permission/Activation errors in PowerShell:**: Run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` and reopen the shell.

**Development & Contributing**
- **Coding style:**: Keep changes focused and minimal. Follow repository's code style.
- **Testing changes:**: Run the relevant script(s) locally and verify output in `extracted_schedule.json` or `ocr_raw.txt`.
- **Pull requests:**: Open PRs against `main` and include a short description and example input/output.

**Contact / Maintainer**
- **Owner:**: `deynklarys` (repository owner). Open issues or PRs for questions or improvements.

--