/*
 * Author: Liangji 
 * Email: liangji20040249@gmail.com
*/
#ifndef CAFFE_SPATIALLSTM_LAYER_HPP_
#define CAFFE_SPATIALLSTM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class SpatialLstmLayer : public Layer<Dtype> {/// Spatial Recurrent Layer, add by liangji, 20150112
public:
  explicit SpatialLstmLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SpatialLstm"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  
  void disorder_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void reorder_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void disorder_gpu_inputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);
  void reorder_gpu_outputdata(const Dtype * datain, Dtype * dataout, bool horizontal, bool reverse,int channels);

  int num_output_;
  int height_out_, width_out_;
  bool bias_term_;
  int num_;
  int channels_;
  int height_, width_;
  int M_;
  int K_;
  int N_;
  int T_;
  int col_count_;
  int col_length_;

  Blob<Dtype> col_buffer_;
  Blob<Dtype> bias_multiplier_;
  Blob<Dtype> data_disorder_buffer_;
  Blob<Dtype> C_buffer_;
  Blob<Dtype> Gate_buffer_;
  Blob<Dtype> H_buffer_;
  Blob<Dtype> FC_1_buffer_;
  Blob<Dtype> identical_multiplier_;
  bool horizontal_;
  bool reverse_;
};



}  // namespace caffe

#endif  // CAFFE_CONCAT_LAYER_HPP_
