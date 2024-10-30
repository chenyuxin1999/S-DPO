import pandas as pd
from movielens_data import MovielensData
from steam_data import SteamData
from goodreads_data import GoodreadsData
from lastfm_data import LastfmData
import json
from tqdm import tqdm


if __name__ == "__main__":
    splits = ["train", "val", "test"]
    cans_num = 20
    data_dir = ""
    for split in splits:
        lastfm = LastfmData(data_dir=data_dir, stage=split, cans_num=cans_num)
        dic_lis = []
        for i in tqdm(range(len(lastfm))):
            dic = {
                "historyList": lastfm[i]["movie_seq"],
                "itemList": lastfm[i]["cans_name"],
                "trueSelection": lastfm[i]["next_title"],
            }
            dic_lis.append(dic)
        with open(f"lastfm-sft-cans20/lastfm-{split}.json", "w") as f:
            json.dump(dic_lis, f, indent=4)

