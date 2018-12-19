import numpy as np
import pywt
from skimage import io, color
from os.path import join
from utils.imresize import imresize
from utils.utils import mkdir_if_not_exist
import h5py


def wmcnn_hdf5(path, full=False):
    def expanddims(array, expand=2):
        for i in range(expand):
            array = np.expand_dims(array, 0)
        return array

    # from matplotlib import pyplot as plt
    # from skimage import measure

    scale = 2
    if not full:
        size_label = 96
        stride = 48
    else:
        size_label = 400

    size_input = size_label / scale
    batch_size = 400
    classes = 7
    # downsizes = [1, 0.7, 0.5]

    id_files = join(path, 'id_files.txt')
    # id_files = join(path, 'test_id.txt')
    with open(id_files, 'r') as f:
        article = f.readlines()
        fnum = len(article)//classes
        order = np.random.permutation(fnum)
        training = order[:int(fnum*0.7)]
        testing = order[len(training):]

        img_list = []
        for i, line in enumerate(article):
            img = io.imread(path+line[2:].strip('\n'))
            if i % fnum in testing:
                cls = line.split('/')[1]
                mkdir_if_not_exist(path+'test/'+cls)
                io.imsave(path+'test/'+line[2:].strip('\n'), img)
            else:
                if img.shape[2] == 4:
                    img = color.rgba2rgb(img)
                img_ycbcr = color.rgb2ycbcr(img) / 255
                (rows, cols, channel) = img_ycbcr.shape
                img_y, img_cb, img_cr = np.split(img_ycbcr, indices_or_sections=channel, axis=2)
                img_list.append(img_y)

    for idx, img_gt in enumerate(img_list):
        if idx == 0:
            hf = h5py.File('D:/data/rsscn7/full_wdata.h5', 'w')
            d_data = hf.create_dataset("data", (batch_size, 1, size_input, size_input), maxshape=(None, 1, size_input, size_input),
                                          dtype='float32')
            d_ca = hf.create_dataset("CA", (batch_size, 1, size_input, size_input), maxshape=(None, 1, size_input, size_input), dtype='float32')
            d_ch = hf.create_dataset("CH", (batch_size, 1, size_input, size_input), maxshape=(None, 1, size_input, size_input), dtype='float32')
            d_cv = hf.create_dataset("CV", (batch_size, 1, size_input, size_input), maxshape=(None, 1, size_input, size_input), dtype='float32')
            d_cd = hf.create_dataset("CD", (batch_size, 1, size_input, size_input), maxshape=(None, 1, size_input, size_input), dtype='float32')
        else:
            hf = h5py.File('D:/data/rsscn7/full_wdata.h5', 'a')
            d_data = hf['data']
            d_ca = hf['CA']
            d_ch = hf['CH']
            d_cv = hf['CV']
            d_cd = hf['CD']

        count = 0
        if not full:
            d_data.resize([idx * batch_size + 392, 1, size_input, size_input])
            d_ca.resize([idx * batch_size + 392, 1, size_input, size_input])
            d_ch.resize([idx * batch_size + 392, 1, size_input, size_input])
            d_cv.resize([idx * batch_size + 392, 1, size_input, size_input])
            d_cd.resize([idx * batch_size + 392, 1, size_input, size_input])

            for flip in range(2):
                for degree in range(4):
                    # for downsize in downsizes:
                    img = img_gt.squeeze()
                    if flip == 1:
                        img = np.fliplr(img)

                    for turn in range(degree):
                        img = np.rot90(img)

                    # img = imresize(img, scalar_scale=downsize)
                    hei, wid = img.shape
                    # fig = plt.figure(figsize=(6, 3))
                    for x in range(0, hei - size_label, stride):
                        for y in range(0, wid - size_label, stride):
                            subim_label = img[x: x + size_label, y: y + size_label]
                            subim_data = imresize(subim_label, scalar_scale=1/scale)
                            coeffs2 = pywt.dwt2(subim_label, 'bior1.1')
                            LL, (LH, HL, HH) = coeffs2

                            d_data[idx * batch_size+count] = expanddims(subim_data, expand=2)
                            d_ca[idx * batch_size+count] = expanddims(LL, expand=2)
                            d_ch[idx * batch_size+count] = expanddims(LH, expand=2)
                            d_cv[idx * batch_size+count] = expanddims(HL, expand=2)
                            d_cd[idx * batch_size+count] = expanddims(HH, expand=2)
                            count += 1
        else:
            d_data.resize([idx * batch_size + 1, 1, size_input, size_input])
            d_ca.resize([idx * batch_size + 1, 1, size_input, size_input])
            d_ch.resize([idx * batch_size + 1, 1, size_input, size_input])
            d_cv.resize([idx * batch_size + 1, 1, size_input, size_input])
            d_cd.resize([idx * batch_size + 1, 1, size_input, size_input])

            img = img_gt.squeeze()
            im_data = imresize(img, scalar_scale=1 / scale)
            coeffs2 = pywt.dwt2(img, 'bior1.1')
            LL, (LH, HL, HH) = coeffs2

            d_data[idx * batch_size + count] = expanddims(im_data, expand=2)
            d_ca[idx * batch_size + count] = expanddims(LL, expand=2)
            d_ch[idx * batch_size + count] = expanddims(LH, expand=2)
            d_cv[idx * batch_size + count] = expanddims(HL, expand=2)
            d_cd[idx * batch_size + count] = expanddims(HH, expand=2)
            count += 1

        batch_size = count

        hf.close()


if __name__=='__main__':
    # expand_hdf5()
    wmcnn_hdf5('../data/RSSCN7/', True)
