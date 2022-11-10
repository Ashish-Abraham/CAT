import os
from PIL import Image


def get_file_paths(folder):
    a_file_paths = []
    b_file_paths = []
    image_file_paths = []
    for root, dirs, filenames in os.walk(folder):
        filenames = sorted(filenames)
        for filename in filenames:
            input_path = os.path.abspath(root)
            file_path = os.path.join(input_path, filename)
            if filename.endswith('.png') or filename.endswith('.jpg'):
                a_file_paths.append(file_path[:len(file_path) - 4]+'c'+".png")
                b_file_paths.append(file_path)
                

        break  # prevent descending into subfolders
    for i in range(len(a_file_paths)):
        a_file_paths[i] = a_file_paths[i].replace("trainB","trainA")

    print(len(b_file_paths),"\n",len(a_file_paths))    

    return a_file_paths, b_file_paths


def align_images(a_file_paths, b_file_paths, target_path):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for i in range(len(a_file_paths)):
        im_size=(256,256)
        img_a = Image.open(a_file_paths[i])
        img_a = img_a.resize(im_size)
        img_b = Image.open(b_file_paths[i])
        img_b = img_b.resize(im_size)
        assert (img_a.size == img_b.size)

        aligned_image = Image.new("RGB", (img_a.size[0] * 2, img_a.size[1]))
        aligned_image.paste(img_a, (0, 0))
        aligned_image.paste(img_b, (img_a.size[0], 0))
        aligned_image.save(os.path.join(target_path, '{:04d}.jpg'.format(i)))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        dest='dataset_path',
        help=
        'Which folder to process (it should have subfolders testA, testB, trainA and trainB'
    )
    args = parser.parse_args()

    dataset_folder = args.dataset_path
    print(dataset_folder)

    test_a_path = os.path.join(dataset_folder, 'valA')
    # test_b_path = os.path.join(dataset_folder, 'testB')
    test_a_file_paths, test_b_file_paths = get_file_paths(test_a_path)
    assert (len(test_a_file_paths) == len(test_b_file_paths))
    test_path = os.path.join(dataset_folder, 'val')

    train_a_path = os.path.join(dataset_folder, 'trainA')
    # train_b_path = os.path.join(dataset_folder, 'trainB')
    train_b_file_paths , train_a_file_paths= get_file_paths(train_a_path)
    assert (len(train_a_file_paths) == len(train_b_file_paths))
    train_path = os.path.join(dataset_folder, 'train')

    for i in range(len(train_b_file_paths)):
        train_b_file_paths[i] = train_b_file_paths[i].replace("trainA","trainB")
        train_b_file_paths[i] = train_b_file_paths[i].replace("cc","")

    for i in range(len(test_b_file_paths)):
        test_b_file_paths[i] = test_b_file_paths[i].replace("valA","valB")
        test_b_file_paths[i] = test_b_file_paths[i].replace("cc","")    

    align_images(test_a_file_paths, test_b_file_paths, test_path)
    align_images(train_a_file_paths, train_b_file_paths, train_path)
