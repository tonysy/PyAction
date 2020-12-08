# Please put cmn split folder, i.e. kinetics-100/[train/val/test].list in this folder

import os

for name in ["train", "val", "test"]:
    f = open("kinetics-100/{}.list".format(name), "r")
    assert f is not None
    lines = f.readlines()
    f.close()

    # gen label
    set_classes = set()
    for line in lines:
        c = line.split("/")[0].replace(" ", "_")
        set_classes.add(c)
    dict_class_label = {k: i for i, k in enumerate(set_classes)}
    print(dict_class_label)

    # gen csv
    with open("{}.csv".format(name), "w") as csv_file:
        for line in lines:
            line = line.replace(" ", "_").rstrip("\n")
            c = line.split("/")[0]
            l = dict_class_label[c]
            csv_file.writelines(
                os.path.expanduser(
                    "~/Datasets/kinetics-400/raw-part/compress/train_256/{}.mp4 {}\n"
                ).format(line, l)
            )

    # verify csv
    with open("{}.csv".format(name), "r") as csv_file:
        lines = csv_file.readlines()
        print("{}: {} lines.".format(name, len(lines)))
