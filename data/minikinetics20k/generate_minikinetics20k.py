import os
import json
from tqdm import tqdm
from collections import defaultdict

# Read full path
full_train_path = "../kinetics/train.csv"
full_val_path = "../kinetics/val.csv"

assert os.path.exists(full_train_path)
assert os.path.exists(full_val_path)

with open(full_train_path, "r") as f:
    full_train_list = f.readlines()

with open(full_val_path, "r") as f:
    full_val_list = f.readlines()

# Read mini kinetics path
mini_train_path = "../minikinetics/train_ytid_list.txt"
mini_val_path = "../minikinetics/val_ytid_list.txt"

assert os.path.exists(mini_train_path)
assert os.path.exists(mini_val_path)

with open(mini_train_path, "r") as f:
    mini_train_list = f.readlines()

with open(mini_val_path, "r") as f:
    mini_val_list = f.readlines()

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

mini_train_list = [item.strip() for item in mini_train_list]
mini_val_list = [item.strip() for item in mini_val_list]

mini_train_out = []
mini_val_out = []

cat_statics_dict = defaultdict(int)
# Get category_mapping
cat_list = []
for item in mini_val_list:
    try:
        cat_list.append(full_val_dict[item]["id"])
    except Exception as e:
        print(e)

cat_list = sorted(list(set(cat_list)))
with open("../kinetics/cat_mapping.json", "r") as f:
    cat_mapping = json.load(f)

cats = sorted(list(cat_mapping.keys()))
id2name = {idx: cats[idx] for idx in range(400)}

mini_cats = sorted([id2name[cat_id] for cat_id in cat_list])
mini_cat2id = {cat_name: idx for idx, cat_name in enumerate(mini_cats)}
mini_id2cat = {idx: cat_name for idx, cat_name in enumerate(mini_cats)}
cat_mapping_mini = {"cat2id": mini_cat2id, "id2cat": mini_id2cat}

if not os.path.exists("./cat_mapping_mini.json"):
    with open("cat_mapping_mini.json", "w") as f:
        json.dump(cat_mapping_mini, f)

train_list_in_dict = defaultdict(list)

for item in tqdm(mini_train_list):
    try:
        path = full_train_dict[item]["path"]
        cat_id = mini_cat2id[id2name[full_train_dict[item]["id"]]]
        cat_statics_dict[cat_id] += 1
        _str = "{} {}\n".format(path, cat_id)
        train_list_in_dict[cat_id].append(_str)

        # mini_train_out.append(_str)
    except Exception as e:
        print(e, "{} not exists".format(item))

for cat_id in train_list_in_dict.keys():
    one_cat_list = sorted(train_list_in_dict[cat_id])[:100]
    mini_train_out += one_cat_list

# import pdb; pdb.set_trace()

for item in tqdm(mini_val_list):
    try:
        path = full_val_dict[item]["path"]
        # cat_id = full_val_dict[item]['id']
        cat_id = mini_cat2id[id2name[full_val_dict[item]["id"]]]
        cat_statics_dict[cat_id] += 1
        _str = "{} {}\n".format(path, cat_id)
        mini_val_out.append(_str)
    except Exception as e:
        print(e, "{} not exists".format(item))

# import pdb;pdb.set_trace()


with open("train.csv", "w") as f:
    f.writelines(mini_train_out)

print("Total Train {} video clips".format(len(mini_train_out)))

with open("val.csv", "w") as f:
    f.writelines(mini_val_out)

with open("test.csv", "w") as f:
    f.writelines(mini_val_out)

print("Total Val {} video clips".format(len(mini_val_out)))
