#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include<math.h>
#include <string>
#include <vector>
#include <fstream>
#include <windows.h>
#include <time.h>
#include <iomanip>
#include <sstream>
#include <device_launch_parameters.h>  //包含blockIdx,threadIdx,gridDim
#include "device_functions.h"
using namespace std;

struct check {
	int edge1;
	int edge2;
	int node;
};
#define THREAD_COUNT 1024 

//我自己的显卡的显存2G,这里先暂时设置2G
#define MAX_MEMORY ((long long)12e9)

//打印获取GPU设备基本信息
void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}

//初始化CUDA
bool InitCUDA()
{
	int count;

	//取得支持Cuda的装置的数目
	cudaGetDeviceCount(&count);

	if (count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;

	for (i = 0; i < count; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		//打印设备信息
		printDeviceProp(prop);
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) {
				break;
			}
		}
	}

	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

/*
先读取成邻接表
再转换成SCR存储结构存储网络
row_ptr:节点表
col_ind:节点邻接表
直接读取到GPU的host显存上
*/
void read_graph(std::string fname, int *&row_ptr, int *&col_ind, int &num_nodes, int &num_edges, bool zero_based = false)
{
	std::ifstream input(fname.c_str());
	if (input.fail())
		throw "No file is found in the current path!";

	// 读网络，这里scr模式下有特殊字符，先处理掉
	std::string line = "%";
	while (line.find("%") != std::string::npos)
	{
		getline(input, line);
	}
	
	std::istringstream ss(line.c_str());
	ss >> num_nodes >> num_nodes >> num_edges;
	int edge_cnt = 0;
	int v1, v2;
	//struct check *c=new check[num_edges];
	//邻接表
	std::vector< std::vector<int> > adj_list(num_nodes);
	for (int i = 0; i < num_edges; i++)
	{
		getline(input, line);
		std::istringstream inp(line.c_str());
		inp >> v1 >> v2;
	/*	c[i].edge1 = v1;
		c[i].edge2 = v2;*/
		if (v1 != v2)
		{
			adj_list[v1].push_back(v2); // add the edge v1->v2
			adj_list[v2].push_back(v1); // add the edge v2->v1
			edge_cnt++;
		}
	}
	input.close();
	num_edges = edge_cnt;


	//gpu host中申请内存
	//+1是因为一开始后面要倒着相减算degree
	cudaMallocHost((void **)&row_ptr, sizeof(int) * (num_nodes + 1));
	cudaMallocHost((void **)&col_ind, sizeof(int) * (2 * num_edges));

	//邻接表转SCR结构
	row_ptr[0] = 0;  //第一格为0
	int index = 0;
	for (int v = 0; v < num_nodes; v++)
	{
		row_ptr[v + 1] = adj_list[v].size();
		for (int i = 0; i < (int)adj_list[v].size(); i++)
		{
			col_ind[index] = adj_list[v][i];
			index++;
		}
	}

	//还需要累加一下才得到最终的row_ptr
	for (int v = 1; v < num_nodes + 1; v++)
	{ // cumulative sum
		row_ptr[v] += row_ptr[v - 1];
	}

}

__global__
void cent_kernel(double *results,  int *dist,  double *sigma,  double *delta, int *rp, int *ci, int n) {//rp:row_ptr ci:cow_idx
	//共享内存变量
	__shared__ int level;
	__shared__ int visited;
	__shared__ double dist_sum;
	//__shared__ double dist_sum_2;
	for (int s = blockIdx.x; s < n; s += gridDim.x) {

		if (threadIdx.x == 0) {
			//results[s] = rp[s + 1] - rp[s]; // 计算度中心性 degree 1 
			level = 0;
			dist_sum = 0.000;
			visited = 1;
		//	dist_sum_2 = 0.0;
			dist[blockIdx.x * n + s] = 0;
			sigma[blockIdx.x * n + s] = 1; //sigma需要提前赋值为1
		}

		__syncthreads();

		// BFS，先宽搜寻找dist
		while (visited == 1) {
			if (threadIdx.x == 0) visited = 0;
			for (int node = threadIdx.x; node < n; node += blockDim.x) {
				for (int edge = rp[node]; edge < rp[node + 1]; edge++) { //相当于遍历cd_idx
					int &adj = ci[edge]; //取别名

					if (dist[(blockIdx.x * n) + adj] == level && dist[(blockIdx.x * n) + node] == -1){
						dist[(blockIdx.x * n) + node] = level + 1;
						visited = 1;
					}

				    if (dist[(blockIdx.x * n) + adj] == level && dist[(blockIdx.x * n) + node] == level + 1) {				
						
						atomicAdd(&sigma[(blockIdx.x * n) + node],sigma[(blockIdx.x * n) + adj]);

					
					}

				}
			
			}
			__syncthreads();
			if (threadIdx.x == 0) level++;   //level相当于brandes里的d,后面用来每步减1
			__syncthreads();
		}
		
	
	
		int dist2_cnt = 0;
	
		// 距离累加
		if (threadIdx.x == 0) {
			for (int i = 0; i < n; i++) {
				if (dist[(blockIdx.x * n) + i] > 0) {					

					dist_sum += dist[(blockIdx.x * n) + i];
					
				}
		
			}
				results[2 * n + s] = (n-1)/dist_sum; //   计算closenes
		}
		
		
		//倒序计算betweeness
		while (level >0) {
			for (int node = threadIdx.x; node < n; node += blockDim.x) {
				if (dist[blockIdx.x * n + node] == level) {   //从距离最远的点开始从后往前
					for (int edge = rp[node]; edge < rp[node + 1]; edge++) {
						int adj = ci[edge];
						if (dist[(blockIdx.x * n) + adj] + 1 == dist[(blockIdx.x * n) + node]) {
							//betweeness，原子加，避免同步操作
							atomicAdd(&delta[(blockIdx.x * n) + adj], (sigma[(blockIdx.x * n) + adj] * 1.0) / sigma[(blockIdx.x * n) + node] * (1 + delta[(blockIdx.x * n) + node]));
						}
					}
				 	atomicAdd(&results[3 * n + node], delta[(blockIdx.x * n) + node]);
				}
	
			}
			__syncthreads();
			if (threadIdx.x == 0) level--;
			__syncthreads();
		}

		//每次重置三个数组的值，以便下一个节点用
		for (int i = 0; i < n; i++) {
			dist[(blockIdx.x * n) + i] = -1;
			sigma[(blockIdx.x * n) + i] = 0;
			delta[(blockIdx.x * n) + i] = 0;
		}
		__syncthreads();
	}

}





double* compute_centralities(int *rp, int *ci, int n, float &time_taken) {
	printf("node_number: %d \n", n);
	/*
		block_count 的设置其实没有特别准确的公式，一般来说看你的任务量是多少，比如100000个点，而你的每个block里面有THREAD_COUNT数量为1024,100000/1024就可以
	*/

	const int BLOCK_COUNT =512; //经验值，一般为128 或 256

	printf("block_count 有块数为:%d \n", BLOCK_COUNT);
	double *sigma;
	int  *dist;
	double *delta;
	double *d_results;

	cudaMalloc((void **)&d_results, sizeof(double) * n * 4); //4类中心性，n个节点
	cudaMalloc((void **)&sigma, sizeof(double  ) * n * BLOCK_COUNT);  //*BLOCK_COUNT,每个block中单独运行
	cudaMalloc((void **)&dist, sizeof( int) * n * BLOCK_COUNT); //
	cudaMalloc((void **)&delta, sizeof(double) * n * BLOCK_COUNT);

	cudaMemset(dist, -1, sizeof( int) * n * BLOCK_COUNT);//初始化
	cudaMemset(sigma, 0, sizeof(double ) * n * BLOCK_COUNT);
	cudaMemset(delta, 0, sizeof(double) * n * BLOCK_COUNT);
	cudaMemset(d_results, 0, sizeof(double) * 4 * n);

	cudaEvent_t start, end; //记录事件
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);

   cent_kernel << <BLOCK_COUNT, THREAD_COUNT >> > (d_results, dist, sigma, delta, rp, ci, n);
	//cent_kernel_seq << <BLOCK_COUNT, THREAD_COUNT >> > (d_results, dist, sigma, delta, rp, ci, n);
	cudaDeviceSynchronize(); //先同步，再时间测量

	cudaEventRecord(end);
	cudaEventSynchronize(end);//先同步，再时间测量

	cudaEventElapsedTime(&time_taken, start, end);//记录gpu时间


	double *results;

	cudaMallocHost((void **)&results, sizeof(double) * n * 4);
	cudaMemcpy(results, d_results, sizeof(double) * n * 4, cudaMemcpyDeviceToHost);

	cudaFree(sigma);
	cudaFree(dist);
	cudaFree(delta);
	cudaFree(d_results);


	cudaDeviceSynchronize();//所有东西都同步，在返回值
	return results;
}


int main()
{
	//CUDA 初始化，查看GPU各项配置
	if (!InitCUDA()) {
		return 0;
	}

	/*
	数据格式检查：1.数据是否存在孤立点，如有先进行处理 2.数据的第一行是否符合输入 node_num>>node_num>>edge_num
	运行前检查：（1）读入文件名替换  （2） 写入文件名替换
	*/
	
	std::string filename = "D:/殷思源/data/musae_facebook_edges-22470-171002.txt";
	int *row_ptr, *col_ind;
	int num_nodes, num_edges;
	read_graph(filename, row_ptr, col_ind, num_nodes, num_edges);
	printf("Graph is read: %s.\n", filename.c_str());

	printf("done scr storage!!!!!\n");
	for (int j = 0; j < num_nodes + 1; j++)
	{
		//printf("row_ptr: %d \n", row_ptr[j]);
	}

	float time_taken;
	int *rp;
	int *ci;
	
	cudaMalloc((void **)&rp, sizeof(int) * (num_nodes + 1));
	cudaMalloc((void **)&ci, sizeof(int) * (num_edges * 2));

	printf("CUDA memory parameters are allocated for kernel function.\n");

	//SCR存储结构-->从host设备复制到device设备上
	cudaMemcpy(rp, row_ptr, sizeof(int) * (num_nodes + 1), cudaMemcpyHostToDevice);
	cudaMemcpy(ci, col_ind, sizeof(int) * (num_edges * 2), cudaMemcpyHostToDevice);
	printf("CUDA memory parameters are set for kernel function.\n");

	double *results = compute_centralities(rp, ci, num_nodes, time_taken);

	printf("Kernel function is finished.核函数执行完成。\n");

	printf("Centrality Results:\n");
	printf("------------------------------------------------------------------------\n");
	ofstream write;
	write.open("D:\\殷思源\\result\\BC\\com-amazon\\com-amazon.ungraph-334863-bc-v2.txt");
	ofstream write1;
	write1.open("D:\\殷思源\\result\\CC\\com-amazon\\com-amazon.ungraph-334863-cc-v2.txt");
	//double nom = (num_nodes - 1)*(num_nodes - 2);
	for (int i = 0; i < num_nodes; i++)
	{
		// printf("%.5f; %.5f; %.5f; %.5f\n ", results[i], results[num_nodes + i], results[2 * num_nodes + i], results[3 * num_nodes + i]);
	
	  //  write << fixed << setprecision(2) << results[3 * num_nodes + i]/2<< endl;
		//write1 << fixed << setprecision(6) << results[2 * num_nodes + i] << endl;
		
	 cout << i<<": "<<" cc:" << results[2 * num_nodes + i]<<"    bc: "<<fixed << setprecision(5)<< results[3 * num_nodes + i]/2 << endl;
	// cout << fixed << setprecision(3) << results[2 * num_nodes + i] << endl;
			//cout << i << ": " << " cc:" << results[2 * num_nodes + i] << endl;
		
	

	}
	write.close();

	printf("the time use of Kernel function: %.5f ms\n", time_taken);

	cudaFreeHost(results);
	cudaFreeHost(row_ptr);
	cudaFreeHost(col_ind);
	cudaFree(rp);
	cudaFree(ci);

	system("pause");
	return 0;
}