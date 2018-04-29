from window_search import *
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from collections import deque


class Tracker():

    def __init__(self):
        '''
        num_frames -- number of frames storing
        heatmaps -- list of previous heatmaps
        output_frames -- list of previous output frames
        '''
        self.num_frames = 20
        self.heatmaps = deque([], maxlen=self.num_frames)
        self.outputs_frames = deque([], maxlen=self.num_frames)


    def detect(self, img, heatmap, threshold=3):
        '''
        detects vehicles in the frame images and update heatmaps history
        '''
        hot_windows = get_labeled_bboxes(np.copy(img), label(heatmap), False)
        # step 1 create new heatmap based on windows found
        new_ave = np.zeros_like(heatmap)
        wmargin, hmargin = 10, 20
        for box in hot_windows:
            new_ave[box[0][1]-wmargin:box[1][1]+wmargin, box[0][0]-hmargin:box[1][0]+hmargin] = 5

        self.heatmaps.append(new_ave)
        ave = sum(self.heatmaps) * 0.2 if len(self.heatmaps) <10 else sum(self.heatmaps) // len(self.heatmaps)

        ave[ave <= threshold] = 0

        # step 2 filter
        self.outputs_frames.append(ave)
        new_heatmap = np.zeros_like(heatmap)
        windows = get_labeled_bboxes(np.copy(img), label(ave), False)
        for box in hot_windows:
            new_heatmap[box[0][1]-wmargin:box[1][1]+wmargin, box[0][0]-hmargin:box[1][0]+hmargin] = 10

        final = sum(self.outputs_frames) * 0.5
        final[final <= 1] = 0

        return final

    def pipeline(self, img):

        # load model and scaler
        scaler = joblib.load('scaler.pkl')
        clf = joblib.load('model.pkl')

        # extract all windows that are detected containing a car with repsective confidence scores
        windows, confids= find_cars(img, ystart=400, ystop=620, scales=np.arange(1,2,0.5), svc=clf, X_scaler=scaler, cells_per_step=1, orient=11, pix_per_cell=16, cell_per_block=2, spatial_size=(32,32), hist_bins=32)

        # get heatmap
        heatmap = get_heatmap(img, windows, confids, 3)
        # update heatmap history
        ave = self.detect(img, heatmap, 4)
        # draw detections on image
        draw, _ = get_labeled_bboxes(np.copy(img), label(ave))

        return draw

if __name__ == "__main__":

    pass
