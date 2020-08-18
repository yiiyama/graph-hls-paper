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
#include <iostream>

#include "myproject.h"
#include "parameters.h"

void myproject(
    input_t input_1[N_INPUT_1_1*N_INPUT_2_1], input2_t input_2[N_INPUT_1_2],
    layer8_t layer8_out[N_LAYER_8], result_t layer11_out[N_LAYER_10],
    unsigned short &const_size_in_1, unsigned short &const_size_in_2,
    unsigned short &const_size_out_1, unsigned short &const_size_out_2
) {

    //hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_PARTITION variable=input_1 cyclic factor=16 dim=0
    #pragma HLS ARRAY_RESHAPE variable=input_2 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,input_2,layer8_out,layer11_out 
    #pragma HLS DATAFLOW 

    const_size_in_1 = N_INPUT_1_1*N_INPUT_2_1;
    const_size_in_2 = N_INPUT_1_2;
    const_size_out_1 = N_LAYER_8;
    const_size_out_2 = N_LAYER_10;

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        //hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<input_transform_0_weights3_t, 128>(input_transform_0_w3, "input_transform_0_w3.txt");
        nnet::load_weights_from_txt<input_transform_0_biases3_t, 32>(input_transform_0_b3, "input_transform_0_b3.txt");
        nnet::load_weights_from_txt<aggregator_distance_0_weights3_t, 16>(aggregator_distance_0_w3, "aggregator_distance_0_w3.txt");
        nnet::load_weights_from_txt<aggregator_distance_0_biases3_t, 4>(aggregator_distance_0_b3, "aggregator_distance_0_b3.txt");
        nnet::load_weights_from_txt<output_transform_0_biases3_t, 8>(output_transform_0_b3, "output_transform_0_b3.txt");
        nnet::load_weights_from_txt<input_transform_1_weights3_t, 256>(input_transform_1_w3, "input_transform_1_w3.txt");
        nnet::load_weights_from_txt<input_transform_1_biases3_t, 32>(input_transform_1_b3, "input_transform_1_b3.txt");
        nnet::load_weights_from_txt<aggregator_distance_1_weights3_t, 32>(aggregator_distance_1_w3, "aggregator_distance_1_w3.txt");
        nnet::load_weights_from_txt<aggregator_distance_1_biases3_t, 4>(aggregator_distance_1_b3, "aggregator_distance_1_b3.txt");
        nnet::load_weights_from_txt<output_transform_1_biases3_t, 8>(output_transform_1_b3, "output_transform_1_b3.txt");
        nnet::load_weights_from_txt<input_transform_2_weights3_t, 1024>(input_transform_2_w3, "input_transform_2_w3.txt");
        nnet::load_weights_from_txt<input_transform_2_biases3_t, 128>(input_transform_2_b3, "input_transform_2_b3.txt");
        nnet::load_weights_from_txt<aggregator_distance_2_weights3_t, 64>(aggregator_distance_2_w3, "aggregator_distance_2_w3.txt");
        nnet::load_weights_from_txt<aggregator_distance_2_biases3_t, 8>(aggregator_distance_2_b3, "aggregator_distance_2_b3.txt");
        nnet::load_weights_from_txt<output_transform_2_biases3_t, 16>(output_transform_2_b3, "output_transform_2_b3.txt");
        nnet::load_weights_from_txt<model_default_t, 256>(w4, "w4.txt");
        nnet::load_weights_from_txt<model_default_t, 16>(b4, "b4.txt");
        nnet::load_weights_from_txt<model_default_t, 128>(w6, "w6.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(b6, "b6.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(w8, "w8.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b8, "b8.txt");
        nnet::load_weights_from_txt<model_default_t, 8>(w10, "w10.txt");
        nnet::load_weights_from_txt<model_default_t, 1>(b10, "b10.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    //hls-fpga-machine-learning insert layers

    layer3_t layer3_out[OUT_FEATURES_3];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0
    nnet::garnet_stack<input_t, input2_t, layer3_t, config3>(input_1, input_2, layer3_out);

    layer4_t layer4_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::dense_latency<layer3_t, layer4_t, config4>(layer3_out, layer4_out, w4, b4);

    layer5_t layer5_out[N_LAYER_4];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0
    nnet::relu<layer4_t, layer5_t, relu_config5>(layer4_out, layer5_out);

    layer6_t layer6_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::dense_latency<layer5_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6);

    layer7_t layer7_out[N_LAYER_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::relu<layer6_t, layer7_t, relu_config7>(layer6_out, layer7_out);

    nnet::dense_latency<layer7_t, layer8_t, config8>(layer7_out, layer8_out, w8, b8);

    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense_latency<layer7_t, layer10_t, config10>(layer7_out, layer10_out, w10, b10);

    nnet::sigmoid<layer10_t, result_t, sigmoid_config11>(layer10_out, layer11_out);

}
