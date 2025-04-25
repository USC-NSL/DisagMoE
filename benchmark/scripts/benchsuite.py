#!/usr/bin/env python3
import requests
import json
import sys

test_cases = {
    "short": {
        "input": [30, 70],
        "output": [70, 130],
        "rate": list(range(100, 500+1, 100)),
    },
    "medium": {
        "input": [50, 150],
        "output": [50, 250],
        "rate": list(range(50, 250+1, 50)),
    },
    "reasonable": {
        "input": [100, 300],
        "output": [100, 500],
        "rate": list(range(25, 100+1, 25)),
    },
    "long": {
        "input": [800, 1200],
        "output": [1600, 2400],
        "rate": list(range(20, 120, 20)),
    },
}

def sanity_check():
    _ = requests.post(
        "http://localhost:6699/run_once",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "rate": 1,
            "time": 1,
            "distribution": "poisson"
        })
    )

def run_one_suite(duration, suite):

    sanity_check()
    
    input_len = test_cases[suite]["input"]
    output_len = test_cases[suite]["output"]
    print(f"Input length: {input_len}, Output length: {output_len}")
    
    for rate in test_cases[suite]["rate"]:
        print(f"Running {suite} with rate {rate} and duration {duration}")
        response = requests.post(
            "http://localhost:6699/run_once",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "rate": rate,
                "time": duration,
                "distribution": "poisson",
                "min_input_len": input_len[0],
                "max_input_len": input_len[1],
                "min_output_len": output_len[0],
                "max_output_len": output_len[1]
            })
        )
    
        print(response.text)
    
    
def main():
    suite = sys.argv[1]
    run_one_suite(80, suite)
    
if __name__ == "__main__":
    main()