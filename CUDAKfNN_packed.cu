/*******************************************************************
*   CUDAKfNN_packed.cu
*   CUDAKfNN_packed
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Oct 21, 2016
*******************************************************************/
//
// Fastest GPU implementation of a brute-force
// matcher for 128-float descriptors such as SIFT
// in 2NN mode, i.e., a match is returned if the best
// match between a query vector and a training vector
// is more than a certain threshold ratio
// better than the second-best match.
//
// Float descriptors are slow. Check out my CUDAK2NN project
// for much faster binary description matching. Use a
// good binary descriptor such as LATCH where possible.
//
// That said, this laboriously crafted kernel is EXTREMELY fast
// for a float matcher.
//
// CUDA CC 3.0 or higher is required.
//
// All functionality is contained in the files CUDAKfNN_packed.h
// and CUDAKfNN_packed.cu. 'main.cpp' is simply a sample test harness
// with example usage and performance testing.
//

#include "CUDAKfNN_packed.h"

// 1.603

__global__ void
#ifndef __INTELLISENSE__
__launch_bounds__(256, 0)
#endif
CUDAKfNN_packed_kernel(const cudaTextureObject_t tex_q, const int num_q, const cudaTextureObject_t tex_t, const int num_t, int* const __restrict__ g_match, const float threshold) {
	int ofs_t = threadIdx.x & 7;
	uint4 train = tex1Dfetch<uint4>(tex_t, ofs_t);
	ofs_t += 8;
	uint4 q[2];
	for (int i = 0, ofs_q = ((threadIdx.x & 24) << 1) + (threadIdx.x & 7) + (blockIdx.x << 9) + (threadIdx.y << 6); i < 2; ++i, ofs_q += 8) q[i] = tex1Dfetch<uint4>(tex_q, ofs_q);
	int best_i;
	uint32_t best_v = 1000000000, second_v = 2000000000;
#pragma unroll 7
	for (int t = 0; t < num_t; ++t, ofs_t += 8) {
		uint32_t dist[2];
#pragma unroll
		for (int i = 0; i < 2; ++i) {
			uint32_t diffs = __vabsdiffu4(q[i].w, train.w);
			uint32_t tmp = __byte_perm(0U, diffs, 4U);
			dist[i] = tmp * tmp;
			tmp = __byte_perm(0U, diffs, 5U);
			dist[i] += tmp * tmp;
			tmp = (diffs >> 16) & 0xFF;
			dist[i] += tmp * tmp;
			tmp = (diffs >> 24) & 0xFF;
			dist[i] += tmp * tmp;

			diffs = __vabsdiffu4(q[i].x, train.x);
			tmp = __byte_perm(0U, diffs, 4U);
			dist[i] += tmp * tmp;
			tmp = __byte_perm(0U, diffs, 5U);
			dist[i] += tmp * tmp;
			tmp = (diffs >> 16) & 0xFF;
			dist[i] += tmp * tmp;
			tmp = (diffs >> 24) & 0xFF;
			dist[i] += tmp * tmp;

			diffs = __vabsdiffu4(q[i].y, train.y);
			tmp = __byte_perm(0U, diffs, 4U);
			dist[i] += tmp * tmp;
			tmp = __byte_perm(0U, diffs, 5U);
			dist[i] += tmp * tmp;
			tmp = (diffs >> 16) & 0xFF;
			dist[i] += tmp * tmp;
			tmp = (diffs >> 24) & 0xFF;
			dist[i] += tmp * tmp;

			diffs = __vabsdiffu4(q[i].z, train.z);
			tmp = __byte_perm(0U, diffs, 4U);
			dist[i] += tmp * tmp;
			tmp = __byte_perm(0U, diffs, 5U);
			dist[i] += tmp * tmp;
			tmp = (diffs >> 16) & 0xFF;
			dist[i] += tmp * tmp;
			tmp = (diffs >> 24) & 0xFF;
			dist[i] += tmp * tmp;
		}
		for (int i = 0; i < 2; ++i) dist[i] += __shfl_xor(dist[i], 1);
		train = tex1Dfetch<uint4>(tex_t, ofs_t);
		if (threadIdx.x & 1) dist[0] = dist[1];
		dist[0] += __shfl_xor(dist[0], 2);
		second_v = min(dist[0] += __shfl_xor(dist[0], 4), second_v);
		if (dist[0] < best_v) {
			second_v = best_v;
			best_i = t;
			best_v = dist[0];
		}
	}
	const int idx = (blockIdx.x << 6) + (threadIdx.y << 3) + ((threadIdx.x & 24) >> 2) + (threadIdx.x & 7);
	if (idx < num_q && ((threadIdx.x & 6) == 0)) g_match[idx] = static_cast<float>(best_v) > threshold * static_cast<float>(second_v) ? -1 : best_i;
}

void CUDAKfNN_packed(const cudaTextureObject_t tex_t, const int num_t, const cudaTextureObject_t tex_q, const int num_q, int* const __restrict d_m, const float threshold) {
	CUDAKfNN_packed_kernel<<<((num_q - 1) >> 6) + 1, { 32, 8 }>>>(tex_q, num_q, tex_t, num_t, d_m, threshold*threshold);
	cudaDeviceSynchronize();
}