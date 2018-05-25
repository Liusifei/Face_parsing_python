/*
File name: weightwrap_layer.cpp
by Guangyu Zhong
(guangyuzhonghikari@gmail.com)
Date: 02/06/2018

Example:
Input feat: N*32*H*W feature map
      left: -1
      top: -1
      output: wrap(feat to 1, 1) dot feat, means each node dot the left top one, i.e., the distance btw each node and its left top node.
      N * 32 * H * W
*/
#include <vector>
//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/filler.hpp"
//#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
//#include "caffe/util/thread_functions.hpp"
#include "caffe/layers/weightwrap_layer.hpp"
//#include "caffe/util/io.hpp"

namespace caffe {

    #define min(a,b) ((a<b)?(a):(b))
    #define max(a,b) ((a>b)?(a):(b))
    // sifeiliu: add a flag
    __device__ int get_translate_index(int index, int channels,int height, int width, int horizontal, int vertical){
        int n = index / (channels * height * width);
        int c = (index - n * (channels * height * width)) / (height * width);
        int h = (index - n * (channels * height * width) - c * (height * width)) / width;
        int w = index - n * (channels * height * width) - c * (height * width) - h * width;

        w = w + horizontal;
        h = h + vertical;

        if ((w >= 0 && w < width) && (h >=0 && h < height))
        	return (n * channels * height * width + c * height * width + h * width + w); //((n * channels + c) * height + h) * width + w.
        else
            return -1;
    }



    template <typename Dtype>
    __global__ void zero_blob(Dtype* input, const int count){
        CUDA_KERNEL_LOOP(index, count){
            input[index] = 0;
        }
    }

    /* sifeiliu: 
        1. we need to avoid duplicated boundaries so that thoes pixels can copy from the prior. 2. change output to num * height * width.
        3. output is inited as zeros.
    */
    template <typename Dtype>
    __global__ void forward_wrap_dot_matrix(const Dtype* input, Dtype* output, int count, int channels,int height, int width, int horizontal, int vertical){
        CUDA_KERNEL_LOOP(index, count){
            int new_index = get_translate_index(index, channels, height, width, horizontal, vertical);
            // int index_reduce = get_translate_channel(index, num, channels, height, width);
            if (new_index!=-1)
                output[index] = input[new_index] * input[index];
        }
    }


    /* sifeiliu: 
        1. revise as the same;
        2. X_diff is inited as zeros;
        //(const Dtype *, const Dtype *, int, int, int, int, int, int) [with Dtype=float]
    */
    template <typename Dtype>
    __global__ void backward_wrap_dot_matrix(const Dtype* W_diff, Dtype* X1_diff, Dtype* X2_diff, const Dtype* X, int count, int channels,int height, int width, int horizontal, int vertical){
        
        //int num = count / (channels * height * width);

        CUDA_KERNEL_LOOP(index, count){
            int new_index = get_translate_index(index, channels, height, width, horizontal, vertical);
            if (new_index!=-1) {    
                // int index_reduce = get_translate_channel(index, num, channels, height, width);
                X1_diff[index] = X[new_index] * W_diff[index];
                X2_diff[new_index] = X[index] * W_diff[index];
            }
        }
    }


    /* sifeiliu:
        1. revise the size of trans
        2. trans init to zero
    */
    template <typename Dtype>
    void WeightWrapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
        const int count = bottom[0]->count();
    	const Dtype* X = bottom[0]->gpu_data();

    	Dtype* trans = top[0]->mutable_gpu_data();
        zero_blob<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(trans, count);
        CUDA_POST_KERNEL_CHECK;

        const int width = bottom[0]->width();
        const int height = bottom[0]->height();
        const int channels = bottom[0]->channels();
        const int num = bottom[0]->num();
        const int horizontal = horizontal_;
        const int vertical = vertical_;
        forward_wrap_dot_matrix<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(X, trans, count, channels, height, width, horizontal, vertical);
        CUDA_POST_KERNEL_CHECK;
    }

    
    template <typename Dtype>
    void WeightWrapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
        const int count = bottom[0]->count();
    	const Dtype* X = bottom[0]->gpu_data(); // source feature map
        const Dtype* W = top[0]->gpu_data();
        const Dtype* W_diff = top[0]->gpu_diff();
        Dtype* X_diff = bottom[0]->mutable_gpu_diff();
        const int width = bottom[0]->width();
        const int height = bottom[0]->height();
        const int channels = bottom[0]->channels();
        const int num = bottom[0]->num();
        const int horizontal = horizontal_;
        const int vertical = vertical_;

        Blob<Dtype> x1_diff(num, channels, height, width);
        Blob<Dtype> x2_diff(num, channels, height, width);
        Dtype* X1_diff = x1_diff.mutable_gpu_diff();
        Dtype* X2_diff = x2_diff.mutable_gpu_diff();

        backward_wrap_dot_matrix<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(W_diff, X1_diff, X2_diff, X, count, channels, height, width, horizontal, vertical);
        CUDA_POST_KERNEL_CHECK;
        caffe_gpu_add(count, x1_diff.gpu_diff(), x2_diff.gpu_diff(), X1_diff);
    }

INSTANTIATE_LAYER_GPU_FUNCS(WeightWrapLayer);
}
