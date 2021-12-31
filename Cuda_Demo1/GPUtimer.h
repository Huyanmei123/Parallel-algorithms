#include <cuda_runtime_api.h>
#include <cuda.h>
class GPUTimer {
public:
	cudaEvent_t time_start;
	cudaEvent_t time_stop;

	GPUTimer() {
		cudaEventCreate(&time_start);
		cudaEventCreate(&time_stop);
	}

	void start() {
		cudaEventRecord(time_start, 0);
	}

	void stop() {
		cudaEventRecord(time_stop, 0);
	}

	double elapsed() {
		float elapsed;
		cudaEventSynchronize(time_stop);
		cudaEventElapsedTime(&elapsed, time_start, time_stop);
		return elapsed;
	}

	~GPUTimer() {
		cudaEventDestroy(time_start);
		cudaEventDestroy(time_stop);
	}

};

