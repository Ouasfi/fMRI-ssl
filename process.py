import numpy as np
import argparse
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Processing fmri data')
    parser.add_argument('-f','--filepath',help = "file to process", type=str, default= 'fmri_97voxels.npz')
    file = parser.parse_args().filepath
    vx = np.load('data/fmri/raw/' + file, allow_pickle =True)['X']
    np.save('data/fmri/processed/sub_1.npy', vx)