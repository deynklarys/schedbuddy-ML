from pathlib import Path

LABEL_PATH = Path("output/1f4a47d1-242/labels/1f4a47d1-242.txt")

with LABEL_PATH.open("r", encoding="utf-8") as file:
    lines = [line.strip() for line in file if line.strip()]

print(f"# of lines: {len(lines)}")
print(lines[0])
