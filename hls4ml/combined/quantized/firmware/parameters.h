#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <complex>
#include "ap_int.h"
#include "ap_fixed.h"

#include "nnet_utils/nnet_helpers.h"
//hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_large.h"
#include "nnet_utils/nnet_garnet.h"
#include "nnet_utils/nnet_garnet_unsigned.h"
 
//hls-fpga-machine-learning insert weights
#include "weights/input_transform_0_w3.h"
#include "weights/input_transform_0_b3.h"
#include "weights/aggregator_distance_0_w3.h"
#include "weights/aggregator_distance_0_b3.h"
#include "weights/output_transform_0_b3.h"
#include "weights/input_transform_1_w3.h"
#include "weights/input_transform_1_b3.h"
#include "weights/aggregator_distance_1_w3.h"
#include "weights/aggregator_distance_1_b3.h"
#include "weights/output_transform_1_b3.h"
#include "weights/input_transform_2_w3.h"
#include "weights/input_transform_2_b3.h"
#include "weights/aggregator_distance_2_w3.h"
#include "weights/aggregator_distance_2_b3.h"
#include "weights/output_transform_2_b3.h"
#include "weights/w4.h"
#include "weights/b4.h"
#include "weights/w6.h"
#include "weights/b6.h"
#include "weights/w8.h"
#include "weights/b8.h"
#include "weights/w10.h"
#include "weights/b10.h"

//hls-fpga-machine-learning insert layer-config
struct config3_base : nnet::garnet_config {
    static const unsigned n_vertices = 128;
    static const unsigned n_vertices_width = 7;
    static const unsigned n_in_features = 4;
    static const unsigned n_in_ufeatures = 0;
    static const unsigned n_in_sfeatures = n_in_features - n_in_ufeatures;
    static const unsigned distance_width = 12;
    static const unsigned output_collapse = collapse_mean;
    static const bool mean_by_nvert = false;

    typedef ap_ufixed<14, 4> norm_t;
    typedef ap_fixed<12, 4, AP_TRN, AP_SAT> distance_t;
    typedef ap_ufixed<10, 0> edge_weight_t;
    typedef ap_ufixed<15, 5> edge_weight_aggr_t;
    typedef ap_fixed<18, 8> aggr_t;
    typedef ap_ufixed<18, 8> uaggr_t;
    typedef layer3_t output_t;

    static const unsigned reuse_factor = 32;
    static const unsigned log2_reuse_factor = 5;

    static const bool is_stack = true;

    typedef config3_base base_t;
};

struct config3 : config3_base {
    static const unsigned n_sublayers = 3;

    template<int L>
    struct sublayer_t : config3_base {};
};

template<>
struct config3::sublayer_t<2> : config3_base {
    static const unsigned n_in_features = 8;
    static const unsigned n_propagate = 16;
    static const unsigned n_aggregators = 8;
    static const unsigned n_out_features = 16;
    static const unsigned n_in_ufeatures = 0;
    static const unsigned n_in_sfeatures = n_in_features - n_in_ufeatures;

    typedef input_transform_2_weights3_t input_transform_weights_t;
    typedef input_transform_2_biases3_t input_transform_biases_t;
    typedef aggregator_distance_2_weights3_t aggregator_distance_weights_t;
    typedef aggregator_distance_2_biases3_t aggregator_distance_biases_t;
    typedef output_transform_2_biases3_t output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[1024];
    static const input_transform_biases_t (&input_transform_biases)[128];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[64];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[8];
    static const output_transform_biases_t (&output_transform_biases)[16];

    typedef config3::sublayer_t<0> next_layer_t;
};

const config3::sublayer_t<2>::input_transform_weights_t (&config3::sublayer_t<2>::input_transform_weights)[1024] = input_transform_2_w3;
const config3::sublayer_t<2>::input_transform_biases_t (&config3::sublayer_t<2>::input_transform_biases)[128] = input_transform_2_b3;
const config3::sublayer_t<2>::aggregator_distance_weights_t (&config3::sublayer_t<2>::aggregator_distance_weights)[64] = aggregator_distance_2_w3;
const config3::sublayer_t<2>::aggregator_distance_biases_t (&config3::sublayer_t<2>::aggregator_distance_biases)[8] = aggregator_distance_2_b3;
const config3::sublayer_t<2>::output_transform_biases_t (&config3::sublayer_t<2>::output_transform_biases)[16] = output_transform_2_b3;

template<>
struct config3::sublayer_t<1> : config3_base {
    static const unsigned n_in_features = 8;
    static const unsigned n_propagate = 8;
    static const unsigned n_aggregators = 4;
    static const unsigned n_out_features = 8;
    static const unsigned n_in_ufeatures = 0;
    static const unsigned n_in_sfeatures = n_in_features - n_in_ufeatures;

    typedef input_transform_1_weights3_t input_transform_weights_t;
    typedef input_transform_1_biases3_t input_transform_biases_t;
    typedef aggregator_distance_1_weights3_t aggregator_distance_weights_t;
    typedef aggregator_distance_1_biases3_t aggregator_distance_biases_t;
    typedef output_transform_1_biases3_t output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[256];
    static const input_transform_biases_t (&input_transform_biases)[32];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[32];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[4];
    static const output_transform_biases_t (&output_transform_biases)[8];

    typedef config3::sublayer_t<2> next_layer_t;
};

const config3::sublayer_t<1>::input_transform_weights_t (&config3::sublayer_t<1>::input_transform_weights)[256] = input_transform_1_w3;
const config3::sublayer_t<1>::input_transform_biases_t (&config3::sublayer_t<1>::input_transform_biases)[32] = input_transform_1_b3;
const config3::sublayer_t<1>::aggregator_distance_weights_t (&config3::sublayer_t<1>::aggregator_distance_weights)[32] = aggregator_distance_1_w3;
const config3::sublayer_t<1>::aggregator_distance_biases_t (&config3::sublayer_t<1>::aggregator_distance_biases)[4] = aggregator_distance_1_b3;
const config3::sublayer_t<1>::output_transform_biases_t (&config3::sublayer_t<1>::output_transform_biases)[8] = output_transform_1_b3;

template<>
struct config3::sublayer_t<0> : config3_base {
    static const unsigned n_in_features = 4;
    static const unsigned n_propagate = 8;
    static const unsigned n_aggregators = 4;
    static const unsigned n_out_features = 8;
    static const unsigned n_in_ufeatures = 0;
    static const unsigned n_in_sfeatures = n_in_features - n_in_ufeatures;

    typedef input_transform_0_weights3_t input_transform_weights_t;
    typedef input_transform_0_biases3_t input_transform_biases_t;
    typedef aggregator_distance_0_weights3_t aggregator_distance_weights_t;
    typedef aggregator_distance_0_biases3_t aggregator_distance_biases_t;
    typedef output_transform_0_biases3_t output_transform_biases_t;

    static const input_transform_weights_t (&input_transform_weights)[128];
    static const input_transform_biases_t (&input_transform_biases)[32];
    static const aggregator_distance_weights_t (&aggregator_distance_weights)[16];
    static const aggregator_distance_biases_t (&aggregator_distance_biases)[4];
    static const output_transform_biases_t (&output_transform_biases)[8];

    typedef config3::sublayer_t<1> next_layer_t;
};

const config3::sublayer_t<0>::input_transform_weights_t (&config3::sublayer_t<0>::input_transform_weights)[128] = input_transform_0_w3;
const config3::sublayer_t<0>::input_transform_biases_t (&config3::sublayer_t<0>::input_transform_biases)[32] = input_transform_0_b3;
const config3::sublayer_t<0>::aggregator_distance_weights_t (&config3::sublayer_t<0>::aggregator_distance_weights)[16] = aggregator_distance_0_w3;
const config3::sublayer_t<0>::aggregator_distance_biases_t (&config3::sublayer_t<0>::aggregator_distance_biases)[4] = aggregator_distance_0_b3;
const config3::sublayer_t<0>::output_transform_biases_t (&config3::sublayer_t<0>::output_transform_biases)[8] = output_transform_0_b3;


struct config4 : nnet::dense_config {
    static const unsigned n_in = OUT_FEATURES_3;
    static const unsigned n_out = N_LAYER_4;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 256;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18, 8> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config5 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

struct config6 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_4;
    static const unsigned n_out = N_LAYER_6;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 128;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18, 8> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};

struct config8 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_out = N_LAYER_8;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 8;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18, 8> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct config10 : nnet::dense_config {
    static const unsigned n_in = N_LAYER_6;
    static const unsigned n_out = N_LAYER_10;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 0;
    static const unsigned n_nonzeros = 8;
    static const bool store_weights_in_bram = false;
    typedef ap_fixed<18, 8> accum_t;
    typedef model_default_t bias_t;
    typedef model_default_t weight_t;
    typedef ap_uint<1> index_t;
};

struct sigmoid_config11 : nnet::activ_config {
    static const unsigned n_in = N_LAYER_10;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef ap_fixed<18,8> table_t;
};


#endif
