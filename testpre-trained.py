import sys
import os
caffe_root='D:/caffe/caffe-master/'
sys.path.append(caffe_root+'python')
import caffe
caffe.set_mode_gpu()
from pylab import *
model_def = 'D:/caffe/caffe-master/models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = 'D:/caffe/caffe-master/examples/imagenet/mydataB/trainA_iter_20000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
def convert_mean(binMean,npyMean):
    blob = caffe.proto.caffe_pb2.BlobProto()
    bin_mean = open(binMean, 'rb' ).read()
    blob.ParseFromString(bin_mean)
    arr = np.array( caffe.io.blobproto_to_array(blob) )
    npy_mean = arr[0]
    np.save(npyMean, npy_mean )
binMean='D:/caffe/caffe-master/examples/imagenet/mydataB/imagenet_mean.binaryproto'
npyMean='D:/caffe/caffe-master/examples/imagenet/mean.npy'
convert_mean(binMean,npyMean)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(npyMean).mean(1).mean(1))
transformer.set_raw_scale('data', 255)  # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2, 1, 0))  # swap channels from RGB to BGR

with open('D:/caffe/caffe-master/examples/imagenet/trainvalB0.txt') as image_list:
    with open('D:/caffe/caffe-master/examples/imagenet/predictionnwall.txt','w') as result:
        count_right=0
        count_all=0
        while 1:
            list_name=image_list.readline()
            if list_name == '\n' or list_name == '':
                break
            image_type=list_name[0:-3].split('.')[-1]
            filename=list_name.split()[0]
            if image_type == 'gif':
                continue
            image = caffe.io.load_image('F:/testB/'+filename)
            #imshow(image)
            transformed_image = transformer.preprocess('data', image)

            net.blobs['data'].data[...] = transformed_image
            net.blobs['data'].reshape(1, 3, 227, 227)
            ### perform classification
            output = net.forward()
            output_prob = net.blobs['prob'].data[0]
            true_label = int(list_name.split(" ")[1])
            predict_label=int(output_prob.argmax())
            if(predict_label==true_label):
                count_right=count_right+1
            count_all=count_all+1
            result.writelines(list_name[0:-1]+' '+str(predict_label)+'\n')
            # if(count_all%100==0):
            print (count_right)
            print (count_all)
        print ('Accuracy: '+ str(float(count_right)/float(count_all)))
        print ('count_all: ' + str(count_all))
        print ('count_right: ' + str(count_right))
        print ('count_wrong: ' + str(count_all-count_right))