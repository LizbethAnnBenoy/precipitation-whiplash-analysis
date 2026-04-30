import json

def save_summary(summary):
    with open('analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
