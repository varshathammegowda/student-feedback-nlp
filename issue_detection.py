issues = {
    "wifi": ["wifi", "internet", "network", "connection"],
    "faculty": ["faculty", "professor", "teacher", "lecturer"],
    "classroom": ["classroom", "class", "lecture hall"],
    "laboratory": ["laboratory", "lab", "equipment"],
    "hostel": ["hostel", "room", "accommodation"],
    "canteen": ["canteen", "food", "mess"],
    "examination": ["exam", "examination", "test", "assessment"],
    "library": ["library", "books", "reading"],
}

def detect_issue(text):
    text = text.lower()

    for issue, keywords in issues.items():
        for keyword in keywords:
            if keyword in text:
                return issue

    return "Other"

text = "The hostel wifi is extremely slow"

issue = detect_issue(text)

print("Feedback:", text)
print("Detected Issue:", issue)