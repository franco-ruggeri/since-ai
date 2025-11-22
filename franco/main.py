import pandas as pd
import json
from semantic_clustering import SemanticClusterer


def load_data():
    with open("data/data_english.json", "r") as f:
        obj = json.load(f)
    section = obj["Stairways"]
    prompt = section["prompt"]
    df = pd.DataFrame(section["data"])
    return prompt, df


def main():
    _, df = load_data()
    clusterer = SemanticClusterer()
    df["text"] = df["Title"] + ". " + df["Observation"]
    df["category"] = clusterer.fit_predict(df["text"])

    print(df.head())
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
