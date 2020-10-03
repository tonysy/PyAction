import os

folder1 = "kinetics-100"
folder2 = "kinetics-100-replaced"

for name in ["train", "val", "test"]:  # ["train"]:
    f1 = open("{}/{}.list".format(folder1, name), "r")
    assert f1 is not None
    lines1 = f1.readlines()
    f1.close()

    f2 = open("{}/{}.list".format(folder2, name), "r")
    assert f2 is not None
    lines2 = f2.readlines()
    f2.close()

    assert len(lines1) == len(lines2)

    l = len(lines1)
    n_replaced = 0
    for i in range(l):
        if lines1[i] != lines2[i]:
            n_replaced += 1
    
    print("split:", name)
    print("ratio | n_replaced | n_lines: {} = {} / {}\n".format(n_replaced/l, n_replaced, l))

