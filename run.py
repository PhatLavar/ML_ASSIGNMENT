import sys
from EDA.eda import main

"""
How to run the EDA process: python run.py <step>
- For full EDA: python run.py eda_full
- For Step 0 (Extract dataset): python run.py eda_step_0
- For Step 1 (Generate basic overview): python run.py eda_step_1
- For Step 2 (Generate integrity check): python run.py eda_step_2
"""

def run_step(step):
    main(step)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <step>")
        sys.exit(1)

    step = sys.argv[1]
    run_step(step)