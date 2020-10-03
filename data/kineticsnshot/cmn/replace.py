import os

def replace(name):
    f = open("kinetics-100/{}.list".format(name), "r")
    assert f is not None
    lines = f.readlines()
    f.close()

    fn_listed = dict()
    fn_candidate = dict()
    
    # Get fn dict
    for line in lines:
        line = line.replace(" ", "_").rstrip("\n")
        c, fn = line.split("/")
        if c in fn_listed:
            fn_listed[c] += fn
        else:
            fn_listed[c] = [fn]

    # Get candidate fn
    for c in fn_listed:
        cpath = os.path.expanduser("~/Datasets/kinetics-400/raw-part/compress/train_256/{}").format(c)
        fn_exist = [p.split(".")[0] for p in os.listdir(cpath)]
        fn_candidate[c] = list(set(fn_exist) - set(fn_listed[c]))
        assert(len(fn_candidate) < len(fn_exist))

    # Update list
    lines_ = []
    for line in lines:
        line = line.replace(" ", "_").rstrip("\n")
        c, fn = line.split("/")
        path = os.path.expanduser("~/Datasets/kinetics-400/raw-part/compress/train_256/{}.mp4").format(line)
        if not os.path.exists(path):
            # replace
            fn_ = fn_candidate[c].pop()
            lines_ += c.replace("_", " ") + "/" + fn_ + "\n"
        else:
            lines_ += c.replace("_", " ") + "/" + fn + "\n"

    # Write new list file
    with open("kinetics-100-replaced/{}.list".format(name), "w") as new_list:
        new_list.writelines(lines_)

for name in ["train", "val", "test"]:  # ["train"]:
    replace(name)
