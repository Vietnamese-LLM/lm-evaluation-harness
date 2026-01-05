import datasets

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.filter(lambda x: x["answer"] in ["A", "B", "C", "D"])

