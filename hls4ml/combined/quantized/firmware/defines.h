#ifndef DEFINES_H_
#define DEFINES_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

//hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 128
#define N_INPUT_2_1 4
#define N_INPUT_1_2 1
#define OUT_FEATURES_3 16
#define N_LAYER_4 16
#define N_LAYER_6 8
#define N_LAYER_8 1
#define N_LAYER_10 1

//hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16, 6> model_default_t;
typedef ap_fixed<14, 5, AP_RND, AP_SAT> input_t;
typedef ap_uint<10> input2_t;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> GarNetStack_default_t;
typedef ap_int<4> input_transform_0_weights3_t;
typedef ap_int<3> input_transform_0_biases3_t;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> aggregator_distance_0_weights3_t;
typedef ap_fixed<12, 2, AP_RND, AP_SAT> aggregator_distance_0_biases3_t;
typedef ap_fixed<11, 1, AP_RND, AP_SAT> output_transform_0_biases3_t;
typedef ap_int<4> input_transform_1_weights3_t;
typedef ap_int<4> input_transform_1_biases3_t;
typedef ap_fixed<13, 3, AP_RND, AP_SAT> aggregator_distance_1_weights3_t;
typedef ap_fixed<11, 1, AP_RND, AP_SAT> aggregator_distance_1_biases3_t;
typedef ap_fixed<11, 1, AP_RND, AP_SAT> output_transform_1_biases3_t;
typedef ap_int<5> input_transform_2_weights3_t;
typedef ap_int<4> input_transform_2_biases3_t;
typedef ap_fixed<13, 3, AP_RND, AP_SAT> aggregator_distance_2_weights3_t;
typedef ap_fixed<11, 1, AP_RND, AP_SAT> aggregator_distance_2_biases3_t;
typedef ap_fixed<11, 1, AP_RND, AP_SAT> output_transform_2_biases3_t;
typedef ap_fixed<16, 6> layer3_t;
typedef ap_fixed<18, 8> Dense_accum_t;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> layer4_t;
typedef ap_fixed<16, 6> layer5_t;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> layer6_t;
typedef ap_fixed<16, 6> layer7_t;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> layer8_t;
typedef ap_fixed<16, 6, AP_RND, AP_SAT> layer10_t;
typedef ap_fixed<16, 6> result_t;

#endif
