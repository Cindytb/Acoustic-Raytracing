#include "convolve.cuh"
namespace cufft {
	// Define the device pointer to the callback routine. The host code will fetch this and pass it to CUFFT
	// Complex multiplication
	__device__ __host__ inline Complex ComplexMulInPlace(Complex a, Complex b) {
		Complex c;
		c.x = a.x * b.x - a.y * b.y;
		c.y = a.x * b.y + a.y * b.x;
		return c;
	}
	__device__ __host__ inline Complex ComplexMulOutPlace(Complex a, Complex b, Complex &c) {
		c.x = a.x * b.x - a.y * b.y;
		c.y = a.x * b.y + a.y * b.x;
	}
	// Complex pointwise multiplication
	__global__ void ComplexPointwiseMulInPlace(Complex* a, const Complex* b, int size) {
		const int numThreads = blockDim.x * gridDim.x;
		const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		for (int i = threadID; i < size; i += numThreads) {
			a[i] = ComplexMulInPlace(a[i], b[i]);
		}
	}
	__global__ void ComplexPointwiseMulOutPlace(const Complex* a, const Complex* b, Complex* c, int size) {
		const int numThreads = blockDim.x * gridDim.x;
		const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		for (int i = threadID; i < size; i += numThreads) {
			ComplexMulOutPlace(a[i], b[i], c[i]);
		}
	}
	// This is the callback routine. It does complex pointwise multiplication with scaling.
	__device__ cufftComplex cbComplexPointwiseMul(void* dataIn, size_t offset, void* cb_info, void* sharedmem) {
		cufftComplex* filter = (cufftComplex*)cb_info;
		return (cufftComplex)ComplexMulInPlace(((Complex*)dataIn)[offset], filter[offset]);
	}
	

#ifndef WIN64
	__device__ cufftCallbackLoadC myOwnCallbackPtr = cbComplexPointwiseMul;
#endif
	void convolve(float* d_ibuf, float* d_rbuf, long long paddedSize) {
		/*Create forward FFT plan*/
		cufftHandle plan;
		CHECK_CUFFT_ERRORS(cufftCreate(&plan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, paddedSize, CUFFT_R2C, 1));

		/*Create inverse FFT plan*/
		cufftHandle outplan;
		CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, paddedSize, CUFFT_C2R, 1));

		/*Transform Complex Signal*/
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_ibuf, (cufftComplex*)d_ibuf));

		/*Transform Filter Signal*/
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, (cufftReal*)d_rbuf, (cufftComplex*)d_rbuf));

#if defined WIN64 || CB == 0
		/*NO CB VERSION*/
		/*CONVOLUTION*/
		int blockSize = 256;
		int numBlocks = (paddedSize + blockSize - 1) / blockSize;

		ComplexPointwiseMulInPlace << < numBlocks, blockSize >> > ((cufftComplex*)d_ibuf, (cufftComplex*)d_rbuf, paddedSize / 2 + 1);
		getLastCudaError("Kernel execution failed [ ComplexPointwiseMulInPlace]");
		checkCudaErrors(cudaDeviceSynchronize());
#else
		/*Copy over the host copy of callback function*/
		cufftCallbackLoadC hostCopyOfCallbackPtr;
		checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr,
			sizeof(hostCopyOfCallbackPtr)));

		/*Associate the load callback with the plan*/
		CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void**)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX,
			(void**)&d_rbuf));

#endif
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_ibuf, (cufftReal*)d_ibuf));
		checkCudaErrors(cufftDestroy(plan));
		checkCudaErrors(cufftDestroy(outplan));
	}

	//Scaling real arrays
	__global__ void RealFloatScale(float* a, long long size, float scale) {
		int numThreads = blockDim.x * gridDim.x;
		int threadID = blockIdx.x * blockDim.x + threadIdx.x;
		for (; threadID < size; threadID += numThreads) {
			a[threadID] *= scale;
		}
	}

	/*Functions to find extrema*/
	float DExtrema(float* pointer, long long size) {
		float* d_min, * d_max;
		float min = 0, max = 0;
		/*Convert raw float pointer into a thrust device pointer*/
		thrust::device_ptr<float> thrust_d_signal(pointer);

		thrust::pair < thrust::device_ptr<float>, thrust::device_ptr<float> >pair =
			thrust::minmax_element(thrust::device, thrust_d_signal, thrust_d_signal + size);


		d_min = pair.first.get();
		d_max = pair.second.get();

		checkCudaErrors(cudaMemcpy(&min, d_min, sizeof(float), cudaMemcpyDefault));
		checkCudaErrors(cudaMemcpy(&max, d_max, sizeof(float), cudaMemcpyDefault));

		return std::abs(min) > max ? std::abs(min) : max;
	}
	
	void normalize_fft(float* d_input, size_t padded_size) {
		int blockSize = 256;
		int numBlocks = (padded_size + blockSize - 1) / blockSize;
		RealFloatScale << < blockSize, numBlocks >> > (d_input, padded_size, 1.0 / padded_size);
	}
	void declip(float* d_input, size_t padded_size) {
		float max = DExtrema(d_input, padded_size);
		int blockSize = 256;
		int numBlocks = (padded_size + blockSize - 1) / blockSize;
		RealFloatScale << < blockSize, numBlocks >> > (d_input, padded_size, 1.0 / max);
	}
	void forward_fft_wrapper(float* d_input, size_t padded_size){
		/*Create forward FFT plan*/
		cufftHandle plan;
		CHECK_CUFFT_ERRORS(cufftCreate(&plan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, padded_size, CUFFT_R2C, 1));
		CHECK_CUFFT_ERRORS(cufftExecR2C(plan, d_input, (cufftComplex*)d_input));
		CHECK_CUFFT_ERRORS(cufftDestroy(plan));
	}
	void inverse_fft_wrapper(float* d_input, size_t padded_size){
		cufftHandle plan;
		CHECK_CUFFT_ERRORS(cufftCreate(&plan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&plan, padded_size, CUFFT_C2R, 1));
		CHECK_CUFFT_ERRORS(cufftExecC2R(plan, (cufftComplex*)d_input, d_input));
		CHECK_CUFFT_ERRORS(cufftDestroy(plan));
	}
	void convolve_ifft_wrapper(cufftComplex* d_ibuf, cufftComplex* d_rbuf, size_t padded_size){
		/*Create inverse FFT plan*/
		cufftHandle outplan;
		CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, padded_size, CUFFT_C2R, 1));
#if defined WIN64 || CB == 0
		/*NO CB VERSION*/
		/*CONVOLUTION*/
		int blockSize = 256;
		int numBlocks = (padded_size + blockSize - 1) / blockSize;

		ComplexPointwiseMulInPlace << < numBlocks, blockSize >> > (d_ibuf, d_rbuf, padded_size / 2 + 1);
		getLastCudaError("Kernel execution failed [ ComplexPointwiseMulInPlace]");
		checkCudaErrors(cudaDeviceSynchronize());
#else
		/*Copy over the host copy of callback function*/
		cufftCallbackLoadC hostCopyOfCallbackPtr;
		checkCudaErrors(cudaMemcpyFromSymbol(&hostCopyOfCallbackPtr, myOwnCallbackPtr,
			sizeof(hostCopyOfCallbackPtr)));

		/*Associate the load callback with the plan*/
		CHECK_CUFFT_ERRORS(cufftXtSetCallback(outplan, (void**)&hostCopyOfCallbackPtr, CUFFT_CB_LD_COMPLEX,
			(void**)&d_rbuf));

#endif
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, (cufftComplex*)d_ibuf, (cufftReal*)d_ibuf));
		checkCudaErrors(cufftDestroy(outplan));
	}
	void convolve_ifft_wrapper(cufftComplex* d_ibuf, cufftComplex* d_rbuf, cufftComplex* d_obuf, size_t padded_size){
		/*Create inverse FFT plan*/
		cufftHandle outplan;
		CHECK_CUFFT_ERRORS(cufftCreate(&outplan));
		CHECK_CUFFT_ERRORS(cufftPlan1d(&outplan, padded_size, CUFFT_C2R, 1));

		/*CONVOLUTION*/
		int blockSize = 256;
		int numBlocks = (padded_size + blockSize - 1) / blockSize;

		ComplexPointwiseMulOutPlace << < numBlocks, blockSize >> > (d_ibuf, d_rbuf, d_obuf, padded_size / 2 + 1);
		getLastCudaError("Kernel execution failed [ ComplexPointwiseMulOutPlace]");
		checkCudaErrors(cudaDeviceSynchronize());
		CHECK_CUFFT_ERRORS(cufftExecC2R(outplan, d_obuf, (cufftReal*)d_obuf));
		checkCudaErrors(cufftDestroy(outplan));
	}
}