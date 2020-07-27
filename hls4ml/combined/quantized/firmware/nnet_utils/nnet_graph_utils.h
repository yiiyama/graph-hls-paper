//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_GRAPH_UTILS_H_
#define NNET_GRAPH_UTILS_H_

#include "nnet_common.h"

namespace nnet {

template<class data_T, class CONFIG_T>
class VertexPacker {
public:
  bool pack_vertices(
    data_T const vertices[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    typename CONFIG_T::nvtx_t nvtx,
    data_T packed[CONFIG_T::n_vertices * CONFIG_T::n_in_features],
    typename CONFIG_T::ngrph_t igraph[CONFIG_T::n_vertices]
  )
  {
    #pragma HLS PIPELINE
  
    if (back_ == 0 && nbuffer_ != 0) {
      for (int ivtx = 0; ivtx < CONFIG_T::n_vertices; ++ivtx) {
        if (ivtx == nbuffer_)
          break;
  
        for (int ifeat = 0; ifeat < CONFIG_T::n_in_features; ++ifeat)
          packed[ivtx * CONFIG_T::n_in_features + ifeat] = buffer_[ivtx * CONFIG_T::n_in_features + ifeat];
  
        igraph[ivtx] = 0;
      }
  
      back_ = nbuffer_;
      npacked_ = 1;
      nbuffer_ = 0;
    }
  
    if (nvtx == 0) {
      // flushing the buffer
      for (int ivtx = 0; ivtx < CONFIG_T::n_vertices; ++ivtx) {
        if (ivtx < nbuffer_)
          continue;
  
        igraph[ivtx] = -1;
      }
  
      return true;
    }
    else if (back_ + nvtx > CONFIG_T::n_vertices) {
      // overflow; save the input in buffer and return
  
      for (int ivtx = 0; ivtx < CONFIG_T::n_vertices; ++ivtx) {
        if (ivtx == nvtx)
          break;
  
        for (int ifeat = 0; ifeat < CONFIG_T::n_in_features; ++ifeat)
          buffer_[ivtx * CONFIG_T::n_in_features + ifeat] = vertices[ivtx * CONFIG_T::n_in_features + ifeat];
      }
  
      for (int ivtx = 0; ivtx < CONFIG_T::n_vertices; ++ivtx) {
        if (ivtx < back_)
          continue;
  
        igraph[ivtx] = -1;
      }
  
      nbuffer_ = nvtx;
      back_ = 0;
      npacked_ = 0;
  
      return true;
    }
    else {
      for (int ivtx = 0; ivtx < CONFIG_T::n_vertices; ++ivtx) {
        if (ivtx < back_ || ivtx >= back_ + nvtx)
          continue;
  
        for (int ifeat = 0; ifeat < CONFIG_T::n_in_features; ++ifeat)
          packed[ivtx * CONFIG_T::n_in_features + ifeat] = vertices[(ivtx - back_) * CONFIG_T::n_in_features + ifeat];
  
        igraph[ivtx] = npacked_;
      }
  
      npacked_ += 1;
  
      if (CONFIG_T::n_graphs > 0 && npacked_ == CONFIG_T::n_graphs) {
        for (int ivtx = 0; ivtx < CONFIG_T::n_vertices; ++ivtx) {
          if (ivtx < back_)
            continue;
  
          igraph[ivtx] = -1;
        }
  
        back_ = 0;
        npacked_ = 0;
        return true;
      }
      else {
        back_ += nvtx;
        return false;
      }
    }
  }

private:
  data_T buffer_[CONFIG_T::n_vertices * CONFIG_T::n_in_features]{};
  typename CONFIG_T::nvtx_t nbuffer_{};
  typename CONFIG_T::nvtx_t back_{};
  typename CONFIG_T::ngrph_t npacked_{};
}

}

#endif
