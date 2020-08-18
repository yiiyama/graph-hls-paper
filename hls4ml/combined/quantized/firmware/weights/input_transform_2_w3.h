//Numpy array shape [16, 8, 8]
//Min -8.000000000000
//Max 7.000000000000
//Number of zeros 157

#ifndef INPUT_TRANSFORM_2_W3_H_
#define INPUT_TRANSFORM_2_W3_H_

#ifndef __SYNTHESIS__
input_transform_2_weights3_t input_transform_2_w3[1024];
#else
input_transform_2_weights3_t input_transform_2_w3[1024] = {2, 1, 3, 0, -1, -4, -5, 2, 1, 2, 0, -2, -2, 0, -5, 4, 6, 0, -3, 0, -6, 3, -1, 5, 1, -1, 1, -1, -1, -4, -3, 1, 0, -2, -1, -3, -2, -1, 0, 0, 3, 2, 2, -3, -2, 0, -2, 3, -4, 0, -1, -1, 3, -2, -1, -2, 1, 2, 2, -2, -2, -3, -6, 3, -2, 5, -4, 1, -1, 4, -4, 3, -1, 4, -2, 1, 1, 4, -2, 2, -3, 1, 3, 2, 3, 0, 2, -1, -2, 1, 3, -1, 2, -3, -4, -1, -2, -1, 5, -2, 2, -5, -1, -3, -2, 3, -2, 2, 3, 4, 2, 0, 3, 0, 1, -2, -4, 0, 0, 4, 1, 1, -2, -4, -3, 4, 1, 3, -3, 1, 3, 0, 2, 1, 4, -1, 1, -4, 5, -2, 0, -4, 2, -2, -6, -1, 4, 0, 5, -4, 1, -5, 3, -6, 2, -3, -2, -3, 4, -2, -5, 0, 4, 0, 5, -4, 0, -4, 0, -5, 4, -2, 2, -6, 2, -3, -2, -1, 3, 2, 3, -1, 3, -3, 1, -4, 1, 1, 0, -3, 4, -2, -5, -1, -2, 2, 2, -2, -3, -3, 5, 1, 0, 0, -3, -1, -5, 4, 0, -2, 0, 2, -2, 0, 1, -1, -1, 2, -2, 0, 1, 0, -4, 2, 3, 1, -2, -2, -4, 3, -3, 3, -1, -1, -1, -2, 0, 0, 3, -1, 2, 0, 1, -1, -3, -1, -3, 3, 3, 3, -1, -3, -4, 2, -5, 5, -2, -1, -4, 1, 1, 1, -1, -2, 6, 1, 3, 0, -3, -1, -5, 4, 0, -3, -4, 3, 1, 2, 4, -2, 1, -1, 0, 0, 0, 0, 0, 0, 2, 1, 4, 3, 1, 0, -1, 2, 6, 0, -4, -1, -5, 3, 1, 5, -1, 2, -5, 4, -1, 3, -3, 1, 2, 1, -1, 0, -2, 1, -1, 2, -2, -2, -2, 0, 1, 1, 2, -2, -1, -4, 3, -2, 0, -6, 0, -3, 0, -1, -4, 1, -1, -1, -2, 0, -1, 1, -2, -3, 0, 4, 3, 0, 7, -4, -4, 0, -6, 0, 0, 2, 2, -3, 1, -3, -4, -4, 1, -1, -4, 1, 0, 2, 3, -1, -1, -1, -1, 1, 2, 1, 0, -2, -4, 0, 0, -4, -5, -1, 0, 1, 3, -1, -2, 2, 1, 2, 2, 1, 1, -1, 6, -3, 0, -2, -5, -1, 1, 2, 0, 3, 3, -1, 0, 0, -1, 1, 7, 0, -3, 0, -6, 3, 0, 5, 2, -1, -1, -2, -1, 4, 3, 2, 2, 0, -1, -2, -2, -1, -2, 2, -1, 3, 1, 0, 1, 3, -1, 1, -1, 2, 3, -2, 1, -1, 1, 0, 3, -3, 4, -2, -3, -6, -2, 1, -5, 3, 3, 1, 4, -3, -3, -3, 2, -5, 1, -1, -2, -4, 1, -2, -7, 0, 5, 0, 6, -3, 1, -5, 0, -2, 3, -5, 0, -5, 3, -1, -3, 0, 0, 2, 2, 1, 1, -2, 0, -3, 3, 0, 0, -6, 1, -2, -3, -1, 4, 1, 3, -6, -1, -3, 4, 1, 1, 0, -3, 1, -3, 4, 4, 1, -4, 0, -6, 4, -1, 5, -1, 1, 1, 0, 1, -1, -2, 0, -5, -1, 0, 0, 2, -2, -1, -3, -4, 2, 0, 1, 2, 1, -1, -2, -4, 2, 1, 3, 2, -1, -2, -2, -2, 1, 1, -1, 1, -1, -3, 0, 3, 4, -3, 2, -2, 5, -2, 5, -6, 3, -4, 1, 2, 4, 2, -3, -3, 1, 3, -1, 5, -1, 0, -2, -3, 3, -2, 2, 2, 3, 0, -1, -6, -2, 6, -1, 5, -6, 3, -6, -5, 2, -1, 3, 2, 1, -3, -1, 0, 2, 2, -1, 2, 1, 1, 1, -2, 1, -1, 1, 1, 2, 2, -1, 1, 0, 5, -2, 1, -1, 2, 0, 2, -1, 2, 0, -2, -1, 2, -1, -3, 0, 1, 0, 3, -1, -2, 0, 3, -3, 1, -2, -3, -1, 2, -1, -3, 5, 2, 0, 4, 1, -2, 1, 0, -5, 5, -2, 0, -6, 1, -4, -2, 4, -2, 4, 1, 5, 0, 1, 0, 0, 0, 3, 0, -1, 0, -1, -3, -1, 7, -1, 3, -6, 0, -3, 3, -5, -2, -1, -2, 0, 5, -1, 4, 1, 1, -1, -3, -2, -5, 2, -1, 0, -1, -3, 0, 5, 4, 0, 6, -2, -6, 1, -5, 2, -1, 2, 1, -2, 3, -1, 1, -4, 3, -3, -6, 2, 0, 0, 5, 1, 2, -2, -6, 3, -1, 2, 5, 2, 0, -2, 0, 1, 0, -1, -2, -1, -1, 2, 1, -4, 0, -1, 0, -4, -1, 0, -1, 1, 1, 2, -1, -2, -3, 2, -1, 3, 0, 1, 2, 0, -4, 1, 5, 0, -4, 0, -5, 4, 0, 4, 1, 0, -1, 0, -2, 0, -2, 1, 2, 1, -3, -3, -3, 1, -3, 3, 0, 0, -1, 1, 0, 1, -1, 0, 2, -1, 0, -4, 0, 3, 4, 1, 2, -1, 7, -1, 0, -3, 2, 0, 2, -1, 4, -4, 0, -1, 2, 0, 3, -4, 3, -2, -2, -2, 3, -1, 0, 0, 7, 1, 2, -4, -2, 0, 3, -2, 2, -4, -1, -3, 0, 1, 1, -2, 1, -1, 0, 1, 3, -1, 0, -1, -2, 2, 1, 2, 2, 0, 3, -1, -3, 0, -3, 3, 2, 1, 0, 4, 2, 2, 1, 3, 2, 0, 2, -2, -3, 0, -1, 3, 4, -1, 1, 1, -1, 0, -3, 1, -1, 1, 1, 0, 5, -3, 1, -1, 2, 1, 0, 1, 0, 1, 1, 3, 0, 1, 4, -1, -5, -1, -3, 4, 2, 2, 3, -1, -3, 0, -2, 3, 2, 1, -3, -1, 1, -2, 3, -3, 1, -1, 1, -4, 3, -1, -1, -7, -2, -2, -3, 3, -1, 1, 1, -1, -5, 1, 1, -6, 2, -1, 0, -5, 2, -2, -3, 2, -1, -1, 0, 2, -1, 0, -2, -2, 7, -1, 2, -8, 1, -3, -5, 3, 1, 4, 5, 0, -1, -2, -2, -1, 2, 0, 1, -5, -2, -2};
#endif

#endif