import json
import numpy as np
import os
import tqdm


def kinetics_extraction_acc():
    # Read full path
    full_train_path = "../kinetics/train.csv"
    full_val_path = "../kinetics/val.csv"

    assert os.path.exists(full_train_path)
    assert os.path.exists(full_val_path)

    with open(full_train_path, "r") as f:
        full_train_list = f.readlines()

    with open(full_val_path, "r") as f:
        full_val_list = f.readlines()

    # Create dict
    full_train_list = [item.strip().split(" ") for item in full_train_list]
    full_val_list = [item.strip().split(" ") for item in full_val_list]

    full_train_dict = {
        os.path.basename(item[0]).split(".")[0][:11]: dict(path=item[0], id=int(item[1]))
        for item in full_train_list
    }

    full_val_dict = {
        os.path.basename(item[0]).split(".")[0]: dict(path=item[0], id=int(item[1]))
        for item in full_val_list
    }

    with open("../kinetics/cat_mapping.json", "r") as f:
        cat_mapping = json.load(f)   

    cats = sorted(list(cat_mapping.keys()))
    id2name = {idx: cats[idx] for idx in range(400)}

    with open('acc_per_class.json', 'r') as f:
        acc_dict = json.load(f)

    # cat_list = []
    acc_list = []

    # cat_list = sorted(list(acc_dict.keys()))
    acc_list = [acc_dict[str(idx)] for idx in range(400)]
    acc_list = sorted(acc_list)

    acc2id = {acc_dict[str(idx)]:idx for idx in range(400)}

    select_cat_list = []
    select_acc_list = acc_list[::10] 
    select_cat_list = [id2name[acc2id[acc_num]] for acc_num in select_acc_list]
    select_id_list = [acc2id[acc_num] for acc_num in select_acc_list]

    mini_cats = sorted(select_cat_list)
    mini_ids = sorted(select_id_list)

    mini_train_out = []
    mini_val_out = []
    # mini_test_out = []

    for item in full_train_dict.keys():
        if full_train_dict[item]["id"] in mini_ids:
            try:
                # path = full_train_dict[item]["path"]
                # cat_id = mini_cat2id[id2name[full_train_dict[item]["id"]]]
                # cat_statics_dict[cat_id] += 1
                # _str = "{} {}\n".format(path, cat_id)
                _str = "{}\n".format(item)
                mini_train_out.append(_str)
            except Exception as e:
                print(e, "{} not exists".format(item))

    for item in full_val_dict.keys():
        if full_val_dict[item]["id"] in mini_ids:
            try:
                # path = full_val_dict[item]["path"]
                # cat_id = mini_cat2id[id2name[full_val_dict[item]["id"]]]
                # cat_statics_dict[cat_id] += 1
                # _str = "{} {}\n".format(path, cat_id)
                _str = "{}\n".format(item)
                mini_val_out.append(_str)
            except Exception as e:
                print(e, "{} not exists".format(item))

    # for item in full_val_dict.keys():
    #     if full_val_dict[item]["id"] in mini_ids:
    #         try:
    #             # path = full_val_dict[item]["path"]
    #             # cat_id = mini_cat2id[id2name[full_val_dict[item]["id"]]]
    #             # cat_statics_dict[cat_id] += 1
    #             # _str = "{} {}\n".format(path, cat_id)
    #             _str = "{}\n".format(item)
    #             mini_test_out.append(_str)
    #         except Exception as e:
    #             print(e, "{} not exists".format(item))

    with open("train_ytid_list.txt", "w") as f:
        f.writelines(mini_train_out)

    print("Total Train {} video clips".format(len(mini_train_out)))

    with open("val_ytid_list.txt", "w") as f:
        f.writelines(mini_val_out)

    # with open("test.csv", "w") as f:
    #     f.writelines(mini_val_out)

    print("Total Val {} video clips".format(len(mini_val_out)))


if __name__ == "__main__":
    kinetics_extraction_acc()