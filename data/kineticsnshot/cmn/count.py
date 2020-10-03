import os

folder = "kinetics-100"

for name in ["train", "val", "test"]:  # ["train"]:
    f = open("{}/{}.list".format(folder, name), "r")
    assert f is not None
    lines = f.readlines()
    f.close()

    with open("{}/{}_missing.list".format(folder, name), "w") as f_misssing:
        n_cmn = 0
        n_found = 0
        for line in lines:
            line_ = line.replace(" ", "_").rstrip("\n")
            path = os.path.expanduser("~/Datasets/kinetics-400/raw-part/compress/train_256/{}.mp4").format(line_)
            n_cmn += 1
            if os.path.exists(path):
                n_found += 1
            else:
                f_misssing.writelines(line)

    print("split:", name)
    print("n_cmn:", n_cmn)
    print("n_found:", n_found)
    print("n_miss:", n_cmn - n_found)
    print("found/cmn:", n_found/n_cmn, "\n")
