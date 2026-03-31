from rapidfuzz import fuzz, process

HEADER_NAMES = ["Code", "Subject", "Units", "Class", "Days", "Time", "Room", "Faculty"]

def match_header(extracted_text, min_score=0):
    # -1 score in case no matches
    max_score = -1

    # Return extracted data when no match
    max_name = extracted_text

    for name in HEADER_NAMES:
        # try other methods
        score = fuzz.ratio(extracted_text, name)
        score = fuzz.partial_ratio(extracted_text, name)
        if (score > min_score) & (score > max_score):
            max_name = name
            max_score = score
    return (max_name, max_score)

if __name__ == "__main__":

    trash_text = [
        "Units Crodit Lec Lab", 
        "Bel", 
        "en ea\nCode", 
        "Subject _", 
        "Room -"
    ]

    for text in trash_text:
        header, score = match_header(text, 50)
        print(f"Match: {header} : {score}")