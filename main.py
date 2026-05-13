import subprocess


def main():
    """Run MooVision Pipeline"""
    print("Hello from moovision! \n")
    # This executes processing.py as a standalone process
    preprocessing = subprocess.run(
        ["python", "scripts/preprocessing.py"], capture_output=True, text=True
    )
    baseline = subprocess.run(
        ["python", "scripts/baseline.py"], capture_output=True, text=True
    )
    evaluation = subprocess.run(
        ["python", "scripts/evaluation.py"], capture_output=True, text=True
    )

    if preprocessing.returncode == 0:
        print(preprocessing.stdout)
    else:
        print(f"Error: {preprocessing.stderr}")

    if baseline.returncode == 0:
        print(baseline.stdout)
    else:
        print(f"Error: {baseline.stderr}")

    if evaluation.returncode == 0:
        print(evaluation.stdout)
    else:
        print(f"Error: {evaluation.stderr}")

    print("Moo! ;)")


if __name__ == "__main__":
    main()
