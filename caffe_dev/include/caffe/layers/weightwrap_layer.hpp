/*
File name: weightwrap_layer.hpp
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
#ifndef CAFFE_WEIGHTWRAP_LAYER_HPP_
#define CAFFE_WEIGHTWRAP_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class WeightWrapLayer : public Layer<Dtype> {
public:
  explicit WeightWrapLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WeightWrapLayer"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> H_;

  int num_;
  int height_;
  int width_;
  int channels_;
  int vertical_;
  int horizontal_;
};
}  // namespace caffe
#endif  // CAFFE_WEIGHTWRAP_LAYER_HPP_
