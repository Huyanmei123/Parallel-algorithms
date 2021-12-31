#ifndef KERNEL_H_
#define KERNEL_H_

#include "Node.h"


#define MAX_ITERATIONS 100    //迭代次数
#define ALPHA 0.85            // PR的参数
#define MAX_BLOCK_SIZE 32

namespace GraphRead_PageRank {

	class Kernel {
	private:

		int nodes_count;
		int edges_count;

		Node *d_pages;
		double *d_pages_probs;
		int *d_edges_list;
		double *d_pages_ranks_sum;
		double *d_dangling_probs_sum;

	public:
		//构造函数
		Kernel(int n, int e) : nodes_count(n), edges_count(e), d_pages(NULL), d_pages_probs(NULL), d_edges_list(NULL) {}

		virtual ~Kernel() {}

		void allocate_data(Node *nodes, double *pages_probs, int *edges_list);

		void run_kernel();

		double *get_result();
	};


} 

#endif /* KERNEL_H_ */
