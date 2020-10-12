import numpy as np
import cv2
import os
import sys
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from disiml import VideoSISO
from scipy.signal import resample

def interpolate_np(old_video, n_frames):
    """return array interpolated along time-axis to n_frames spanning the same time"""
    if n_frames != old_video.shape[0]:
        new_shape = (n_frames, old_video.shape[1], old_video.shape[2],1)
        result = np.zeros(new_shape, dtype=np.uint8)

        old_t = np.linspace(0, 1, old_video.shape[0])
        new_t = np.linspace(0, 1, n_frames)

        for n, t in enumerate(new_t):
            if n == 0:
                result[n, :, :] = old_video[0, :, :]
            elif n == (len(new_t) - 1):
                result[n, :, :] = old_video[-1, :, :]
            else:
                idx = np.argmax(old_t > t)
                x0 = old_t[idx - 1]
                x1 = old_t[idx]  # check boundary
                y0 = old_video[idx - 1, :, :]
                y1 = old_video[idx, :, :]  # check boundary
                result[n, :, :] = (y0 * (1 - ((t - x0) / (x1 - x0))) +
                                   y1 * ((t - x0) / (x1 - x0))).astype(np.uint8)
    else:  #Same number of frames
        result = old_video
    return result


def loadvideo(filename: str, fps: int) -> np.ndarray:
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, 109, 150, 1), np.uint8) # size required by ap4 model

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame_width==112 and frame_height==112:
            v[count, :, 19:150-19, 0] = frame[:109,:]
        else:
            frame = cv2.resize(frame, (150,109))
            v[count] = frame

    if fps!=30:
        new_size = int(np.round(frame_count*30/fps))
        v = interpolate_np(v, new_size).astype('uint8')
        # v = resample(v.astype(float), num=new_size, axis=0).astype('uint8')

    if v.shape[0]>60: 
        v = v[:60,...]
    else:
        v2 = np.zeros((60, 109, 150, 1), np.uint8)
        v2[:v.shape[0], ...] = v
        v = v2
    return v


class AVIGenerator(tf.keras.utils.Sequence):
    '''
    File Generator class that assumes that each data sample is named as in <name>.npy and all samples reside in a `data_path` folder. See Sequence class in https://keras.io/utils/. 

    Parameters
    ----------
        data_path:str
            Absolute path to the folder that contians numpy arrays
        batch_size: int
            Number of smaples per batch
        y_set: pd.Series or pd.DataFrame
            Pandas Series/DataFrame indexed by sample name and contains a sample per row    
    Attributes
    ----------
        n_batches: int
            Total number of batches as computed from the number of samples / batch size
        data_shape: tuple
            Shape of a single sample. Note: this is used by the fit_generator to build the keras model.

    '''
    def __init__(self, metadata, batch_size : int):
        self.batch_size = batch_size
        self.data = metadata.copy() # Pandas DataFrame with file names in index

        # internals
        self.n_batches = int(np.ceil(self.data.shape[0] / float(self.batch_size)))
        # Do not load file, just get the shape
        self.data.set_index('FilePath', inplace=True)
        a_sample = self.id2sample(self.data.index[0])
        self.data_shape = list(a_sample.shape)

    def id2sample(self, id):
        sample = loadvideo(id, self.data.loc[id, 'FPS'])
        return sample

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.n_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        ids = self.data.index[index*self.batch_size:(index+1)*self.batch_size]

        # at first run create a buffer to allocate data
        if not hasattr(self, 'buffer'):
            self.buffer = np.zeros([self.batch_size,] + self.data_shape)
        
        for i, id in enumerate(ids):
            self.buffer[i,...] = self.id2sample(id)

        X = self.buffer[:len(ids)]
        y = self.data.loc[ids, 'EF'].values

        return X,y


def plot():
    df =  pd.read_csv('FileList_predicted.csv')
  
    sns.set(style="whitegrid")
    df['EF_Group'] = pd.cut(df.EF,[0,35,50,65,80,100])
    BAR2 = (225/255, 221/255, 191/255)
    BAR3 = (76/255, 131/255, 122/255)
    TCOL = (200/255, 200/255, 200/255)

    fig,ax1 = plt.subplots(1)

    sns.set(style="whitegrid", font='serif')

    sns.boxplot(x="EF_Group", y="Pred", data=df, palette="Set1", ax=ax1, width=.5, whis=1, fliersize=1, notch=True)
    face_color = (210/255, 194/255, 149/255)
    edge_color = (140/255, 21/255, 21/255)
    for i,artist in enumerate(ax1.artists):
        # Set the linecolor on the artist to the facecolor, and set the facecolor to None
        col = artist.get_facecolor()
        artist.set_edgecolor(edge_color)
        artist.set_facecolor(face_color)

        # Each box has 6 associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same colour as above
        for j in range(i*6,i*6+6):
            line = ax1.lines[j]
            line.set_color(edge_color)
            line.set_mfc(edge_color)
            line.set_mec(edge_color)

    # sns.despine(ax=ax1)
    ax1.set_xlabel('Ejection Fraction', fontsize=16)
    ax1.set_ylabel('AP4 Risk Score', fontsize=16)
    ax1.set_ylim(0,1)
    plt.savefig('stanford_risk_vs_ef.pdf',
                bbox_inches="tight",
                # transparent=True,
            )

if __name__=='__main__':

    echonet_path = sys.argv[1]

    data = pd.read_csv(echonet_path + '/FileList.csv')
    data['FilePath'] = data.FileName.apply(lambda x: echonet_path +f'/Videos/{x}')
    
    model_path = f'models/ap4/model'
    if os.path.exists(model_path + '.h5') and \
       os.path.exists(model_path + '.json'):
        print('Loading model %s' % model_path)
        model = VideoSISO(verbose = False)
        model.load(model_path)

    gen = AVIGenerator(data, 100)
    preds= model.predict(gen)
    data['Pred']= preds
    data['BinPred']= preds>0.5
    data.to_csv('FileList_predicted.csv')
    plot()