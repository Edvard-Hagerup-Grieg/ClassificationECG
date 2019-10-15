from statsmodels.robust import mad
from dataset import load_dataset
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pywt
import cv2
import os


folder_path = "C:\\data\\"
pkl_file_name = "data_2033_img_long.pkl"


def load_img_dataset(folder_path=folder_path, pkl_file_name=pkl_file_name):

    if not os.path.exists(folder_path+pkl_file_name):
        print("Converting dataset to images is started. It's take some time.")

        xy = load_dataset(folder_path)
        X = xy["x"]
        Y = xy["y"]

        X_new = []
        Y_new = []
        for i in np.arange(600, 1000, 1):
            print("\rSignal %s/" % str(i + 1) + str(X.shape[0]) + ' is converted.', end='')
            x = X[i, :, 0]
            x = wavelet_smooth(x, wavelet="db4", level=1, title=None)

            peaks_idx = find_peaks_div(x)
            peaks_idx.sort()

            for j in range(len(peaks_idx) - 2):

                start = peaks_idx[j] + (int)((peaks_idx[j + 1] - peaks_idx[j]) / 2)
                finish = peaks_idx[j + 1] + (int)((peaks_idx[j + 2] - peaks_idx[j + 1]) / 2)

                fig = plt.figure(frameon=False)
                plt.plot(x[start:finish], linewidth=4)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                filename = folder_path + 'tmp.png'
                fig.savefig(filename)

                im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.resize(im_gray, (256, 256), interpolation=cv2.INTER_LANCZOS4)

                #X_new.append(im_gray.flatten())
                X_new.append(im_gray)
                Y_new.append(Y[i])

        X_new = np.array(X_new)
        Y_new = np.array(Y_new)

        print("\nThe image dataset is saved.", end='\n')
        print("X shape: ", X_new.shape)
        print("Y shape: ", Y_new.shape)

        outfile = open(folder_path+pkl_file_name, 'wb')
        pkl.dump({"x":X_new, "y":Y_new}, outfile)
        outfile.close()

    infile = open(folder_path + pkl_file_name, 'rb')
    xy = pkl.load(infile)
    infile.close()

    return xy

def load_img_long_dataset(folder_path=folder_path):

    if not os.path.exists(folder_path+pkl_file_name):
        print("Converting dataset to images is started. It's take some time.")

        xy = load_dataset(folder_path)
        X = xy["x"]
        Y = xy["y"]

        X_new = []
        Y_new = []
        for i in range(1000):
            print("\rSignal %s/" % str(i + 1) + str(X.shape[0]) + ' is converted.', end='')
            x = X[i, :, 0]
            x = wavelet_smooth(x, wavelet="db4", level=1, title=None)

            peaks_idx = find_peaks_div(x)
            peaks_idx.sort()

            for j in range(len(peaks_idx) - 3):

                start = peaks_idx[j] + (int)((peaks_idx[j + 1] - peaks_idx[j]) / 2)
                finish = peaks_idx[j + 2] + (int)((peaks_idx[j + 3] - peaks_idx[j + 2]) / 2)

                fig = plt.figure(frameon=False)
                plt.plot(x[start:finish], linewidth=4)
                plt.xticks([]), plt.yticks([])
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)

                filename = folder_path + 'tmp.png'
                fig.savefig(filename)

                im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                im_gray = cv2.resize(im_gray, (512, 256), interpolation=cv2.INTER_LANCZOS4)
                #cv2.imwrite(filename, im_gray)

                #X_new.append(im_gray.flatten())
                X_new.append(im_gray)
                Y_new.append(Y[i])

        X_new = np.array(X_new)
        Y_new = np.array(Y_new)

        print("\nThe image dataset is saved.", end='\n')
        print("X shape: ", X_new.shape)
        print("Y shape: ", Y_new.shape)

        outfile = open(folder_path+pkl_file_name, 'wb')
        pkl.dump({"x":X_new, "y":Y_new}, outfile)
        outfile.close()

    infile = open(folder_path + pkl_file_name, 'rb')
    xy = pkl.load(infile)
    infile.close()

    return xy

def wavelet_smooth(X, wavelet="db4", level=1, title=None):

    coeff = pywt.wavedec(X, wavelet, mode="per")
    sigma = mad(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(X)))

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    y = pywt.waverec(coeff, wavelet, mode="per")

    return y

def find_peaks_div(x, scope_max=10, scope_null=50):

    len = x.shape[0]
    y0 = np.zeros(len)
    y1 = np.zeros(len)
    y2 = np.zeros(len)
    y3 = np.zeros(len)

    for i in range(len-2):
        y0[i+2] = abs(x[i+2]-x[i])
    for i in range(len-4):
        y1[i+4] = abs(x[i+4]-2*x[i+2]+ x[i])
    for i in range(len-4):
        y2[i+4] = 1.3*y0[i+4]+1.1*y1[i+4]
    for i in range(len-4-7):
        for k in range(7):
            y3[i] += y2[i+4-k]
        y3[i] /= 8


    max_idx = []

    curr_max = max(y3)
    curr_argmax = np.argmax(y3)
    true_argmax = np.argmax(x[max(0,curr_argmax-scope_max):min(curr_argmax+scope_max,len)])

    max_idx.append(max(0, curr_argmax-scope_max) + true_argmax)
    y3[max(0,curr_argmax-scope_null):min(curr_argmax+scope_null,len)] *= 0

    prev_max = curr_max
    curr_max = max(y3)

    while (prev_max - curr_max) < (prev_max / 4.0):

        curr_argmax = np.argmax(y3)
        true_argmax = np.argmax(x[max(0,curr_argmax-scope_max):min(curr_argmax+scope_max,len)])

        max_idx.append(max(0, curr_argmax-scope_max) + true_argmax)
        y3[max(0,curr_argmax-scope_null):min(curr_argmax+scope_null,len)] *= 0

        prev_max = curr_max
        curr_max = max(y3)

    return max_idx


if __name__ == "__main__":
    load_img_long_dataset()