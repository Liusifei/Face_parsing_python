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

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/io.hpp"
#include "caffe/layers/weightwrap_layer.hpp"

namespace caffe {
    #define min(a,b) ((a<b)?(a):(b))
    #define max(a,b) ((a>b)?(a):(b))

    template <typename Dtype>
    void WeightWrapLayer<Dtype>::LayerSetUp( const vector<Blob<Dtype>*> & bottom,
        const vector<Blob<Dtype>*> & top ){
            vertical_ = this->layer_param_.weightwrap_param().vertical();
            horizontal_ = this->layer_param_.weightwrap_param().horizontal();
        }

    template <typename Dtype>
    void WeightWrapLayer<Dtype>::Reshape( const vector<Blob<Dtype>*> & bottom,
        const vector<Blob<Dtype>*> & top ){
            CHECK(top.size()==1)<<"top size must equal to 1";
            int bottomsize=bottom.size();
            CHECK(bottom.size()==1)<<"should be input 1 bottoms :(feat). now get "<<bottom.size();

            height_  = bottom[0]->height();
            width_ = bottom[0]->width();
            channels_ = bottom[0]->channels();
            num_ = bottom[0]->num();
            for(int i=0;i<bottomsize;i++){
                CHECK(bottom[i]->num() == num_)<<"all bottom num must equal. "<<bottom[i]->num()<<" vs "<<num_;
                CHECK(bottom[i]->channels() == channels_)<<"all data and gate channels must equal. "<<bottom[i]->channels()<<" vs "<<channels_;
                CHECK(bottom[i]->height() == height_)<<"all bottom height must equal. "<<bottom[i]->height()<<" vs "<<height_;
                CHECK(bottom[i]->width() == width_)<<"all bottom width must equal. "<<bottom[i]->height()<<" vs "<<height_;
            }
            for (int top_id = 0; top_id < top.size(); ++top_id) {
                top[top_id]->Reshape(num_, channels_, height_, width_);
            }
        }
    //
    // template <typename Dtype>
    // void WeightWrapLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*> & bottom,
    //     const vector<Blob<Dtype>*> & top ){
    //         this->Forward_gpu(bottom,top);
    //     }
    template <typename Dtype>
    void WeightWrapLayer<Dtype>::Forward_cpu( const vector<Blob<Dtype>*> & bottom,
        const vector<Blob<Dtype>*> & top ){
            const int width = top[0]->width();
            const int height = top[0]->height();
            const int channels = top[0]->channels();
            const int num = top[0]->num();
            const int horizontal = horizontal_;
            const int vertical = vertical_;
            const int wh_size = width * height;
            const int whc_size = width * height * channels;
            Dtype* output = top[0]->mutable_cpu_data(); // weight matrix
            const Dtype* input = bottom[0]->cpu_data(); // feature map

            for (int n = 0; n<num; n++){
                int off = n * whc_size;
                for (int c=0; c<channels;c++){
                    int off_c = c * wh_size;
                    for (int h=0; h<height; h++){
                        for (int w=0; w < width; w++){
                            int index = off + off_c + h * width + w;
                            int tmp_h = max(h + vertical, 0);
                            tmp_h = min(tmp_h, height-1);
                            int tmp_w = max(w + horizontal, 0);
                            tmp_w = min(tmp_w, width-1);
                            int index_next = off + off_c + tmp_h * width + tmp_w;
                            output[index] = input[index_next] * input[index];
                        }
                    }
                }
            }
        }

    // template <typename Dtype>
    // void WeightWrapLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*> & top,
    //     const vector<bool> & propagate_down,
    //     const vector<Blob<Dtype>*> & bottom ){
    //         this->Backward_gpu(top,propagate_down,bottom);
    //     }
    template <typename Dtype>
    void WeightWrapLayer<Dtype>::Backward_cpu( const vector<Blob<Dtype>*> & top,
        const vector<bool> & propagate_down,
        const vector<Blob<Dtype>*> & bottom ){
            const int width = top[0]->width();
            const int height = top[0]->height();
            const int channels = top[0]->channels();
            const int num = top[0]->num();
            const int horizontal = horizontal_;
            const int vertical = vertical_;
            const int wh_size = width * height;
            const int whc_size = width * height * channels;
            const Dtype* weight_data = top[0]->cpu_data();
            const Dtype* feat_data = bottom[0]->cpu_data();
            const Dtype* weight_diff = top[0]->cpu_diff();
            Dtype* feat_diff = bottom[0]->mutable_cpu_diff();
            for (int n=0; n<num; n++){
                int off = n * whc_size;
                for (int c=0; c<channels;c++){
                    int off_c = c * wh_size;
                    for (int h=0; h<height; h++){
                        for (int w=0; w < width; w++){
                            int index = off + off_c + h * width + w;
                            int tmp_h = max(h + vertical, 0);
                            tmp_h = min(tmp_h, height-1);
                            int tmp_w = max(w + horizontal, 0);
                            tmp_w = min(tmp_w, width-1);
                            int index_next = off + off_c + tmp_h * width + tmp_w;
                            feat_diff[index] = weight_diff[index]*feat_data[index_next];
                        }
                    }
                }
            }
        }

#ifdef CPU_ONLY
STUB_GPU( WeightWrapLayer );
#endif

    INSTANTIATE_CLASS( WeightWrapLayer );
    REGISTER_LAYER_CLASS( WeightWrap );
}  /* namespace caffe */
