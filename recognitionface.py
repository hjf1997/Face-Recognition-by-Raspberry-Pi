
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import facenet
import os
from PIL import Image
import pickle
from scipy import misc
import align.detect_face
from six.moves import xrange
import time

class predict:

    def __init__(self,modelpath, classifier_filename):
        self.sess = tf.Session()
        facenet.load_model(modelpath)
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (self.model, self.class_names) = pickle.load(infile)
        print('Loaded classifier model from file "%s"\n' % classifier_filename_exp)


    def reco_face(self, image_files, image_size):

        # Load the model
        #facenet.load_model(model)

        # Get input and output tensors
        result = []

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        images, count_per_image, nrof_samples, img_cv = self.load_and_align_data(image_files, image_size, margin = 44, gpu_memory_fraction = 1.0)
        if type(images) == int:
            #print("没有识别到人")
            result.append(False)
            return result
        else:
            result.append(True)

        # Run forward pass to calculate embeddings
        feed_dict = {images_placeholder: images, phase_train_placeholder: False}
        emb = self.sess.run(embeddings, feed_dict=feed_dict)
        predictions = self.model.predict_proba(emb)
        best_class_indices = np.argmax(predictions, axis=1)
        for la in range(predictions.shape[0]):
            if predictions[la, best_class_indices[la]] < 0.8:
                best_class_indices[la] = 4
        print(predictions)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        k = 0
        clas_peo = 'detected:'
        for i in range(nrof_samples):
            #print("\npeople in image %s :" % (image_files[i]))
            for j in range(count_per_image[i]):
                result.append(self.class_names[best_class_indices[k]])
                #print('%s: %.3f' % (self.class_names[best_class_indices[k]], best_class_probabilities[k]))
                clas_peo = clas_peo + ' ' + self.class_names[best_class_indices[k]] + ' '
                k += 1
        return result


    def load_and_align_data(self, image_paths, image_size, margin, gpu_memory_fraction):

        minsize = 20 # minimums size of face
        threshold = [ 0.6, 0.7, 0.7 ]  # three teps's threshold
        factor = 0.709 # scale fac  or

        print('Creating networks and loading parameters')
        with tf.Graph().as_default():

            #GPU设定需要更改
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

        nrof_samples = len(image_paths)
        img_list = []
        count_per_image = []
        for i in xrange(nrof_samples):
            #从这里修改传值
            img = misc.imread(os.path.expanduser(image_paths[i]))
            img_size = np.asarray(img.shape)[0:2]
            bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
            count_per_image.append(len(bounding_boxes))
            img_cv = img
            for j in range(len(bounding_boxes)):

                det = np.squeeze(bounding_boxes[j,0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0]-margin/2, 0 )
                bb[1] = np.maximum(det[1]-margin/2, 0 )
                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                cropped = img[bb[1]:bb[3],bb[0]:bb[2], :]
                aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                img_list.append(prewhitened)
        if img_list != []:
            images = np.stack(img_list)
        else:
            images = 0
        return images, count_per_image, nrof_samples,img_cv

if __name__ == '__main__':
    time_start = time.time()
    face = predict(r'E:\Anaconda3\Lib\site-packages\facenet-master\src\models\20170511-185253\20170511-185253.pb', r'E:\Anaconda3\Lib\site-packages\facenet-master\src\my_classifier.pkl')
    time_end = time.time()
    print('加载模型需要' + str(time_end-time_start) + 's')
    for i in range(10):
        time_start = time.time()
        face.reco_face([r'E:\Anaconda3\Lib\site-packages\facenet-master\data\testinblackbox\8.jpg'],160)
        time_end = time.time()
        print('测试一张图片需要' + str(time_end - time_start) + 's')



