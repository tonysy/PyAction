import os

data_prefix = "/public/sist/home/hexm/Datasets/UCF/UCF-101"

train_txt = "/public/sist/home/hexm/Datasets/UCF/ucfTrainTestlist/trainlist01.txt"
test_txt = "/public/sist/home/hexm/Datasets/UCF/ucfTrainTestlist/testlist01.txt"
cat_map_txt = "/public/sist/home/hexm/Datasets/UCF/ucfTrainTestlist/classInd.txt"

with open(cat_map_txt, "r") as f:
    cat_map_str = f.readlines()

cat_map_list = [item.strip().split(" ") for item in cat_map_str]
cat2id_dict = {item[1]: int(item[0]) - 1 for item in cat_map_list}

id2cat_dict = {int(item[0]): item[1] for item in cat_map_list}

with open(train_txt, "r") as f:
    train_str = f.readlines()
train_list = [item.strip().split(" ") for item in train_str]

# Train list
train_out_list = []
for item in train_list:
    path = os.path.join(data_prefix, item[0])
    assert os.path.exists(path)
    train_out_list.append("{} {}\n".format(path, int(item[1]) - 1))


with open(test_txt, "r") as f:
    test_str = f.readlines()
test_list = [item.strip() for item in test_str]

test_out_list = []
for item in test_list:
    cat_name = item.strip().split("/")[0]
    cat_id = cat2id_dict[cat_name]
    path = os.path.join(data_prefix, item)
    assert os.path.exists(path)
    test_out_list.append("{} {}\n".format(path, cat_id))


with open("train.csv", "w") as f:
    f.writelines(train_out_list)

with open("test.csv", "w") as f:
    f.writelines(test_out_list)

with open("val.csv", "w") as f:
    f.writelines(test_out_list)

# import pdb;pdb.set_trace()
