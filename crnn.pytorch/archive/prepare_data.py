from glob import glob

from os.path import join

path = "/Users/rohuntripathi/Downloads/Completed_Transciptions_Filter/"

transcribe_names = glob(path + "*", recursive=True)


for folder in transcribe_names:

    print(join(join(path, folder), "*/*.pdf"))


    image_names = glob(join(join(path, folder), "*/*.pdf"), recursive=True)
    file_names = glob(join(join(path, folder), "*/*/*.docx"), recursive=True)

    image_names = sorted(image_names)
    file_names = sorted(file_names)

    for i in range(min(len(image_names), len(file_names))):
        print(image_names[i] + "_" + file_names[i])



