import cv2
import numpy as np
import SvmTrain as st
# 通过调用OpenCV函数创建ANN
def make_ann(sample_num):
    ann = cv2.ml.ANN_MLP_create()
    ann.setLayerSizes(np.array([64, 1000, 1000, 1000, 8]))
    ann.setTrainMethod(cv2.ml.ANN_MLP_RPROP | cv2.ml.ANN_MLP_UPDATE_WEIGHTS)
    ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
    ann.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1))
    dataset = st.readdata(sample_num)
    responses = np.zeros([8 * sample_num, 8])
    responses = np.float32(responses)
    for i in range(8):
        responses[sample_num * i: sample_num * (i + 1), i] = 1
    ann.train(dataset, cv2.ml.ROW_SAMPLE, responses)
    return ann

if( __name__ == "__main__"):
    ann = make_ann(100)
    testsample = np.load("F:\\traindata\\1\\2.npy")
    testsample = np.float32(testsample)
    l = np.array([testsample])
    result = ann.predict(l)
    print(result)
