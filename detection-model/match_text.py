from rapidfuzz import fuzz, process
import json

HEADER_NAMES = ["code", "subject", "units", "class", "days", "time", "room", "faculty"]

def match_header(extracted_text, min_score=0):
    # -1 score in case no matches
    max_score = -1

    # Return extracted data when no match
    max_name = extracted_text

    for name in HEADER_NAMES:
        # try other methods
        score = max(fuzz.partial_ratio(extracted_text, name), fuzz.ratio(extracted_text, name))
        if (score > min_score) & (score > max_score):
            max_name = name
            max_score = score
    return (max_name, max_score)

def match_course(extracted_text, min_score=0):

    with open('databases/comsci.json', 'r', encoding='utf-8') as f:
        db_dict = json.load(f)
    
    # -1 score in case no matches
    max_score = -1

    # Return extracted data when no match
    max_code = extracted_text

    for code, subject in db_dict.items():
        # try other methods
        score = max(fuzz.partial_ratio(extracted_text, code), fuzz.ratio(extracted_text, code))
        if (score > min_score) & (score > max_score):
            max_code = code
            max_score = score
    return (max_code, max_score, db_dict[max_code])

if __name__ == "__main__":

    trash_text = [
        "ISTP 12", ">S 103", ">S 107", "Nath-102", "SEC 12",
    ]

    for text in trash_text:
        code, subject, score = match_course(text, 50)
        print(f"Match: {code} : {subject}: {score}")
