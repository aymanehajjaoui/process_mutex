/*
 * Copyright (C) 2010-2021 Arm Limited or its affiliates. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* ----------------------------------------------------------------------
 * Project:      CMSIS NN Library
 * Title:        arm_convolve_HWC_q15_basic_nonsquare.c
 * Description:  Q15 version of convolution
 *
 * $Date:        June 17, 2022
 * $Revision:    V.1.0.1
 *
 * Target Processor:  Cortex-M cores
 *
 * -------------------------------------------------------------------- */

#include "arm_nnfunctions.h"
#include "arm_nnsupportfunctions.h"

/**
 *  @ingroup groupNN
 */

/**
 * @addtogroup NNConv
 * @{
 */

/**
 * @brief Basic Q15 convolution function (non-square shape)
 * @param[in]       Im_in       pointer to input tensor
 * @param[in]       dim_im_in_x  input tensor dimention x
 * @param[in]       dim_im_in_y  input tensor dimention y
 * @param[in]       ch_im_in    number of input tensor channels
 * @param[in]       wt          pointer to kernel weights
 * @param[in]       ch_im_out   number of filters, i.e., output tensor channels
 * @param[in]       dim_kernel_x filter kernel size x
 * @param[in]       dim_kernel_y filter kernel size y
 * @param[in]       padding_x    padding size x
 * @param[in]       padding_y    padding size y
 * @param[in]       stride_x     convolution stride x
 * @param[in]       stride_y     convolution stride y
 * @param[in]       bias        pointer to bias
 * @param[in]       bias_shift  amount of left-shift for bias
 * @param[in]       out_shift   amount of right-shift for output
 * @param[in,out]   Im_out      pointer to output tensor
 * @param[in]       dim_im_out_x output tensor dimension x
 * @param[in]       dim_im_out_y output tensor dimension y
 * @param[in,out]   bufferA     pointer to buffer space for input
 * @param[in,out]   bufferB     pointer to buffer space for output
 * @return     The function returns <code>ARM_MATH_SUCCESS</code>
 *
 * @details
 *
 * <b>Buffer size:</b>
 *
 * bufferA size: ch_im_in*dim_kernel_x*dim_kernel_y
 *
 * bufferB size: 0
 *
 * This basic version is designed to work for any input tensor and weight
 * dimension.
 */

arm_status arm_convolve_HWC_q15_basic_nonsquare(const q15_t *Im_in,
                                      const uint16_t dim_im_in_x,
                                      const uint16_t dim_im_in_y,
                                      const uint16_t ch_im_in,
                                      const q15_t *wt,
                                      const uint16_t ch_im_out,
                                      const uint16_t dim_kernel_x,
                                      const uint16_t dim_kernel_y,
                                      const uint16_t padding_x,
                                      const uint16_t padding_y,
                                      const uint16_t stride_x,
                                      const uint16_t stride_y,
                                      const q15_t *bias,
                                      const uint16_t bias_shift,
                                      const uint16_t out_shift,
                                      q15_t *Im_out,
                                      const uint16_t dim_im_out_x,
                                      const uint16_t dim_im_out_y,
                                      q15_t *bufferA,
                                      q7_t *bufferB)
{
    (void)bufferB;
#if defined(ARM_MATH_DSP) && !defined(ARM_MATH_MVEI)
    /* Run the following code for Cortex-M4 and Cortex-M7 */

    int16_t i_out_y, i_out_x, i_ker_y, i_ker_x;

    uint16_t im2col_out_pixel_index = 0;
    q15_t *pBuffer = bufferA;
    q15_t *pOut = Im_out;
    q15_t *im_buffer = bufferA;
    const q15_t *pA;
    int i;

    /* This part implements the im2col function */
    for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
        {
            for (i_ker_y = i_out_y * stride_y - padding_y; i_ker_y < i_out_y * stride_y - padding_y + dim_kernel_y; i_ker_y++)
            {
                for (i_ker_x = i_out_x * stride_x - padding_x; i_ker_x < i_out_x * stride_x - padding_x + dim_kernel_x; i_ker_x++)
                {
                    if (i_ker_y < 0 || i_ker_y >= dim_im_in_y || i_ker_x < 0 || i_ker_x >= dim_im_in_x)
                    {
                        /* Filling 0 for out-of-bound paddings */
                        /* arm_fill_q15(0, pBuffer, ch_im_in); */
                        memset(pBuffer, 0, sizeof(q15_t) * ch_im_in);
                    }
                    else
                    {
                        /* arm_copy_q15((q15_t *) Im_in + (i_ker_y * dim_im_in + i_ker_x) * ch_im_in, pBuffer,
                         * ch_im_in); */
                        memcpy(pBuffer,
                               (q15_t *)Im_in + (i_ker_y * dim_im_in_x + i_ker_x) * ch_im_in,
                               sizeof(q15_t) * ch_im_in);
                    }
                    pBuffer += ch_im_in;
                }
            }

            pA = wt;
            for (i = 0; i < ch_im_out; i++)
            {
                q31_t sum = ((q31_t)bias[i] << bias_shift) + NN_ROUND(out_shift);
                const q15_t *pB = im_buffer;
                uint16_t colCnt = ch_im_in * dim_kernel_y * dim_kernel_x >> 2;
                while (colCnt)
                {
                    q31_t inA1 = arm_nn_read_q15x2_ia(&pA);
                    q31_t inB1 = arm_nn_read_q15x2_ia(&pB);
                    q31_t inA2 = arm_nn_read_q15x2_ia(&pA);
                    q31_t inB2 = arm_nn_read_q15x2_ia(&pB);

                    sum = __SMLAD(inA1, inB1, sum);
                    sum = __SMLAD(inA2, inB2, sum);

                    colCnt--;
                }
                colCnt = ch_im_in * dim_kernel_y * dim_kernel_x & 0x3;
                while (colCnt)
                {
                    q15_t inA1 = *pA++;
                    q15_t inB1 = *pB++;
                    sum += inA1 * inB1;
                    colCnt--;
                }
                *pOut = (q15_t)__SSAT((sum >> out_shift), 16);
                pOut++;
            }

            /* counter reset */
            pBuffer = im_buffer;
            im2col_out_pixel_index++;
        }
    }

#else
    (void)bufferA;
    /* Run the following code as reference implementation for Cortex-M0 and Cortex-M3 */
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
                conv_out = ((q31_t)bias[i] << bias_shift) + NN_ROUND(out_shift);
                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_x; n++)
                    {
                        in_row = stride_y * j + m - padding_y;
                        in_col = stride_x * k + n - padding_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + l] *
                                    wt[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in + l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q15_t)__SSAT((conv_out >> out_shift), 16);
            }
        }
    }

#endif /* ARM_MATH_DSP */

    /* Return to application */
    return ARM_MATH_SUCCESS;
}

/**
 * @} end of NNConv group
 */
