#include <vector>

#include "caffe/layers/huber_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HuberLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  delta_ = this->layer_param_.huber_loss_param().delta();;
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
  error_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
      NOT_IMPLEMENTED;
}

template <typename Dtype>
void HuberLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(HuberLossLayer);
#endif

INSTANTIATE_CLASS(HuberLossLayer);
REGISTER_LAYER_CLASS(HuberLoss);

}  // namespace caffe
