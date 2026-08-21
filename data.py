import gzip
import json

def load_humaneval_dataset(file_path):
    dataset = []
    
    # 'rt' opens the file in text mode, allowing us to read it line by line
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            task = json.loads(line)
            dataset.append(task)
            
    return dataset

# 1. Load the data
humaneval_data = load_humaneval_dataset('HumanEval.jsonl.gz')

# 2. Verify the number of tasks matches the benchmark (164)
print(f"Total tasks loaded: {len(humaneval_data)}\n")

# 3. Look at a specific task (e.g., the first one) to see its structure
first_task = humaneval_data

print(f"Task ID: {first_task.get('task_id')}")
print("-" * 40)

# The 'prompt' contains the function signature and docstring
print("PROMPT (Signature & Docstring):")
print(first_task.get('prompt'))
print("-" * 40)
  
# The 'test' contains the hidden test suite mentioned in the paper
print("HIDDEN TEST SUITE:")
print(first_task.get('test'))
print("-" * 40)