import os.path
import re

import datasets
import tqdm
import ujson


def _save_data_in_json(dataset: datasets.Dataset, save_to: str) -> list[list[str]]:
    data = []
    for row in tqdm.tqdm(dataset):
        chosen = row["chosen"]
        phrases = re.split("\n\nHuman: |\n\nAssistant: ", chosen)
        phrases = phrases[1:-1]
        data.append(phrases)
    dir = os.path.dirname(save_to)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(save_to, "w") as json_file:
        ujson.dump(data, json_file)


def main():
    hh = datasets.load_dataset("Anthropic/hh-rlhf")
    _save_data_in_json(hh["train"], "data/ppo_data/hh/train.json")
    _save_data_in_json(hh["test"], "data/ppo_data/hh/valid.json")


if __name__ == "__main__":
    main()
