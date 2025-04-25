#!/usr/bin/env python3
import requests
import json
import sys

test_cases = {
    "reasonable": {
        "input": [100, 300],
        "output": [100, 500],
        "rate": {
            "top1_gqa": [240, 300], # 40, 60, 80, 120, 160, 200, 
            "top1_mqa": [80, 160, 240, 320, 400], # 20, 40, 
        }
    },
    "reasonable_v2": {
        "input": [50, 150],
        "output": [50, 250],
        "rate": {
            "top1_gqa": [200, 300, 400, 500, 600, 700], # 50, 100, 
            "top1_mqa": [20, 40, 80, 160, 240, 320, 400], # 20, 40, 80, 
        }
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

def run_one_suite(duration, lens, exp):

    sanity_check()
    
    input_len = test_cases[lens]["input"]
    output_len = test_cases[lens]["output"]
    print(f"Input length: {input_len}, Output length: {output_len}")
    
    for rate in test_cases[lens]["rate"][exp]:
        print(f"Running {lens}_{exp} with rate {rate} and duration {duration}")
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
    lens = sys.argv[1]
    exp = sys.argv[2]
    run_one_suite(80, lens, exp)
    
if __name__ == "__main__":
    main()