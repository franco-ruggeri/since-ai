import pandas as pd
import json
from semantic_clustering import SemanticClusterer


def load_data():
    with open("data/data.json", "r") as f:
        obj = json.load(f)
    return pd.DataFrame(obj["Rappuset"]["Data"])


def main():
    df = load_data()
    df.info()

    clusterer = SemanticClusterer()
    df["text"] = df["Heading"] + ". " + df["Observation"]
    df["category"] = clusterer.fit_predict(df["text"])

    print(df.head())
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
