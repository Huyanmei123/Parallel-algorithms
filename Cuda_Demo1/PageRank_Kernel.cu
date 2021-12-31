#include "Kernel.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "GPUTimer.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <device_launch_parameters.h>  
#include "device_functions.h"

namespace GraphRead_PageRank {

	//��ʼ��d_dangling_probs_sum
	__global__ void initialize_dangling_sum(double *d_dangling_probs_sum) {
		d_dangling_probs_sum[0] = 0.0;
	}

	//��ʼ��d_page_ranks_sum
	__global__ void initialize_pages_ranks_sum(double *d_page_ranks_sum) {
		d_page_ranks_sum[0] = 0.0;
	}

	__global__ void calculate_dangling_sum_and_normalize(Node *d_pages, double *d_page_probs, int pages_count, double *d_page_ranks_sum, double *d_dangling_probs_sum) {
		// Calculate page index from thread address
		// ���μ��ÿһ��thread
		int idx = blockIdx.y * blockDim.y + threadIdx.y;

		if (idx < pages_count) {
			d_page_probs[idx] /= d_page_ranks_sum[0];

			//������ǳ���Ϊ0�ĵ�
			if (d_pages[idx].dangling_node) {
				atomicAdd(d_dangling_probs_sum, double(d_page_probs[idx]));
			}
		}
	}

	__global__ void run_page_rank_iteration(Node *d_pages,
		double *d_page_probs,
		int *d_edges_list,  //����in_node
		int pages_count,
		double *d_pages_ranks_sum,
		double *d_dangling_probs_sum,
		double alpha) {

		//ȡ�߳��±�
		int idx = blockIdx.y * blockDim.y + threadIdx.y;

		double new_rank = 0.0;

		if (idx < pages_count) {
			double c_element = 0.0;


			//˭����idx��������d_edges_list�н��в���
			// d_page����ṹ��Ĵ�ŵ�start_idx��end_idx �൱���㷨��һ����cow_ptr
			int i_start = d_pages[idx].Adjstart_index;
			int i_end = d_pages[idx].Adjend_index;

			//���ھӽڵ㴫������ֵ���  
			// �����d_edges_list,���൱���㷨��һ���� col_idx
			for (int i = i_start; i < i_end; i++) {
				int from = d_edges_list[i];
				c_element += d_page_probs[from] / (d_pages[from].links_count);
			}

			//��alhpa���ӽ�������
			new_rank = ((1 - alpha) * 1.0 / pages_count) + (alpha * c_element);// +(alpha * 1.0 / pages_count * d_dangling_probs_sum[0]);  //����汾���������ҽڵ��Ӱ�죬Ҳ���԰��������һ��ɾ��
		}

		//__threadfence();
		__syncthreads();
		if (idx < pages_count) {
		d_page_probs[idx] = new_rank;  //���µ�pagerankֵ����probs���´ε�����
		//atomicExch(&d_page_probs[idx],new_rank);
		atomicAdd(d_pages_ranks_sum, new_rank);
		}
	}

	void Kernel::run_kernel() {
		// Calculate the grid and block sizes.
		// ceil ����ȥ��
		int grid_size = int(ceil(1.0 * nodes_count / MAX_BLOCK_SIZE));
		int block_size = MAX_BLOCK_SIZE;
		GPUTimer timer;
	
		// ��������þ���һ�����У�grid_size��block��ÿ��block����block_size��thread
		if (block_size < 1024) {
			dim3 dimGrid(1, grid_size);//ȷ��grid����block��
			dim3 dimBlock(1, block_size);//ȷ��block����thread��
			GPUTimer gpu_timer1;
			//gpu_timer1.e_start();
			timer.start();
			for (int i = 0; i < 100; ++i) {
				if (i > 0) {
					initialize_dangling_sum << <1, 1 >> > (d_dangling_probs_sum);
					calculate_dangling_sum_and_normalize << <dimGrid, dimBlock >> > (d_pages, d_pages_probs, nodes_count, d_pages_ranks_sum, d_dangling_probs_sum);
				}

				initialize_pages_ranks_sum << <1, 1 >> > (d_pages_ranks_sum);
				run_page_rank_iteration << < dimGrid, dimBlock >> > (d_pages, d_pages_probs, d_edges_list, nodes_count, d_pages_ranks_sum, d_dangling_probs_sum, ALPHA);
			}
			timer.stop();
			//	gpu_timer1.e_stop();
				//cout << "end PageRank time in gpu " << gpu_timer1.elapsed() << " ms." << endl;
		}

		printf("time :%lf\n", timer.elapsed());
		/*else {
			cout << "������һ��block�����ֵ1024" << endl;
		}*/
	}

	void Kernel::allocate_data(Node *pages, double *pages_probs, int *edges_list) {
		double one = 1.0, zero = 0.0;

		// Allocate memory at the gpu device
		cudaMalloc((void **)&d_pages, sizeof(Node) * nodes_count);
		cudaMalloc((void **)&d_pages_probs, sizeof(double) * nodes_count);
		cudaMalloc((void **)&d_edges_list, sizeof(int) * edges_count);
		cudaMalloc((void **)&d_pages_ranks_sum, sizeof(double));  //
		cudaMalloc((void **)&d_dangling_probs_sum, sizeof(double));

		// Copy data from host (cpu) to the gpu
		cudaMemcpy(d_pages, pages, sizeof(Node) * nodes_count, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pages_probs, pages_probs, sizeof(double) * nodes_count, cudaMemcpyHostToDevice);
		cudaMemcpy(d_edges_list, edges_list, sizeof(int) * edges_count, cudaMemcpyHostToDevice);
		cudaMemcpy(d_pages_ranks_sum, &one, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dangling_probs_sum, &zero, sizeof(double), cudaMemcpyHostToDevice);
	}

	double *Kernel::get_result() {
		double *pages_probs = new double[nodes_count];

		cudaMemcpy(pages_probs, d_pages_probs, sizeof(double) * nodes_count, cudaMemcpyDeviceToHost);
		cudaError_t e = cudaGetLastError();
		if (e != cudaSuccess) {
			printf("CUDA����ʧ��  %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
			exit(0);
		}

		return pages_probs;
	}

} /* namespace PageRank */
