import pandas as pd
from movielens_data import MovielensData
from steam_data import SteamData
from goodreads_data import GoodreadsData
from lastfm_data import LastfmData
import json
from tqdm import tqdm


if __name__ == "__main__":
    # splits = ["train", "val", "test"]
    # cans_num = 20
    # data_dir = "../LLaRA_data/LLaRA_ml"
    # for split in splits:
    #     ml = MovielensData(data_dir=data_dir, stage=split, cans_num=cans_num)
    #     dic_lis = []
    #     for i in tqdm(range(len(ml))):
    #         dic = {
    #             "historyList": ml[i]["seq_name"],
    #             "itemList": ml[i]["cans_name"],
    #             "trueSelection": ml[i]["correct_answer"],
    #         }
    #         dic_lis.append(dic)
    #     with open(f"ml-sft-cans20-new/ml-{split}.json", "w") as f:
    #         json.dump(dic_lis, f, indent=4)
    
    # splits = ["train", "val", "test"]
    # cans_num = 20
    # data_dir = "../LLaRA_data/steam"
    # for split in splits:
    #     steam = SteamData(data_dir=data_dir, stage=split, cans_num=cans_num)
    #     dic_lis = []
    #     for i in tqdm(range(len(steam))):
    #         dic = {
    #             "historyList": steam[i]["seq_name"],
    #             "itemList": steam[i]["cans_name"],
    #             "trueSelection": steam[i]["correct_answer"],
    #         }
    #         dic_lis.append(dic)
    #     with open(f"steam-sft-cans20-new/steam-{split}.json", "w") as f:
    #         json.dump(dic_lis, f, indent=4)


    # splits = ["train", "val", "test"]
    # cans_num = 20
    # data_dir = "/storage_fast/yxchen/DPO-Rec/LLaRA_data/goodreads"
    # for split in splits:
    #     steam = GoodreadsData(data_dir=data_dir, stage=split, cans_num=cans_num)
    #     dic_lis = []
    #     for i in tqdm(range(len(steam))):
    #         # print(steam[i])
    #         dic = {
    #             "historyList": steam[i]["movie_seq"],
    #             "itemList": steam[i]["cans_name"],
    #             "trueSelection": steam[i]["next_title"],
    #         }
    #         dic_lis.append(dic)
    #     with open(f"goodread-sft-cans20/goodread-{split}.json", "w") as f:
    #         json.dump(dic_lis, f, indent=4)

    splits = ["train", "val", "test"]
    cans_num = 20
    data_dir = "/storage_fast/yxchen/DPO-Rec/LLaRA_data/lastfm"
    for split in splits:
        steam = LastfmData(data_dir=data_dir, stage=split, cans_num=cans_num)
        dic_lis = []
        for i in tqdm(range(len(steam))):
            # print(steam[i])
            dic = {
                "historyList": steam[i]["movie_seq"],
                "itemList": steam[i]["cans_name"],
                "trueSelection": steam[i]["next_title"],
            }
            dic_lis.append(dic)
        with open(f"lastfm-sft-cans20-new/lastfm-{split}.json", "w") as f:
            json.dump(dic_lis, f, indent=4)

