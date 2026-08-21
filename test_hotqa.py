from datasets import load_dataset

ds = load_dataset("hotpot_qa", "distractor", split="validation")

print(f"Total samples: {len(ds)}")
print(f"Keys: {ds[0].keys()}")
print(f"\nQuestion: {ds[0]['question']}")
print(f"Answer: {ds[0]['answer']}")
print(f"Type: {ds[0]['type']}")
print(f"Level: {ds[0]['level']}")
print(f"Num supporting facts: {len(ds[0]['supporting_facts']['title'])}")
print(f"Num context paragraphs: {len(ds[0]['context']['title'])}")