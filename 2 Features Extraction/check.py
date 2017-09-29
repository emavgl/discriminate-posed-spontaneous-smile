import os, sys, glob, shutil

# argv params
folder = sys.argv[1]

missing = 0

subdirs = [x[0] for x in os.walk(folder)]

for subdir in subdirs:
    search_model = subdir + "/*.lm"

    # get the list of all the lm files
    file_list = glob.glob(search_model)

    if len(file_list) == 0 and subdir != folder:
        print(subdir, "is empty")
        shutil.rmtree(subdir)
        missing += 1

    for i in range(0, len(file_list)):
        file_path = subdir + "/frame" + str(i) + ".jpg.lm"
        ex = os.path.exists(file_path)
        if not ex:
            print("error in ", subdir, file_path, "not exists")
            missing += 1
            shutil.rmtree(subdir)
            break

print("incomplete files")
print(missing)
