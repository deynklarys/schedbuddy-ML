from datetime import datetime

from models import TimeData 

def parse_time(time_str):
    parts = [part.strip() for part in time_str.split("-")]
    if len(parts) != 2:
        raise ValueError("Expected time range like '05:00 PM - 07:00 PM'")

    time_format = "%I:%M %p"
    start_time = datetime.strptime(parts[0], time_format).time()
    start_time_mins = start_time.hour * 60 + start_time.minute
    end_time = datetime.strptime(parts[1], time_format).time()
    end_time_mins = end_time.hour * 60 + end_time.minute
    return TimeData(
        start=start_time,
        end=end_time,
        start_mins=start_time_mins,
        end_mins=end_time_mins,
    )

if __name__ == "__main__":

    time_samples = [
        "05:00 PM - 07:00 PM",
        "01:00 PM - 04:00 PM",
        "10:00 AM - 12:00 PM",
        "08:30 AM - 10:00 AM",
    ]
    for time in time_samples:
        time_data = parse_time(time)
        print(f"Time (24-hour): {time_data.start} - {time_data.end}")
        print(f"Time (minutes): {time_data.start_mins} - {time_data.end_mins}")
        print(f"Duration: {time_data.end_mins - time_data.start_mins}")
