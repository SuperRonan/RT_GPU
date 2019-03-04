#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "include_directory.cuh"



template <class A, class B, class C>
__global__ void cudadd(const A * a, const B * b, C * c)
{
	int i = blockIdx.x;
	c[i] = a[i] + b[i];
}


void fill_random(int * a, int l)
{
	for (int i = 0; i < l; ++i)
	{
		a[i] = rand();
	}
}

template <size_t N, class T>
void fill_vector_array(math::Vector<N, T> * vec, int l)
{
	for (int i = 0; i < l; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			vec[i][j] = i + j;
		}
	}
}

template <class T>
void print_array(const T * a, int l)
{
	std::cout << "----------------------" << std::endl;
	for (int i = 0; i < l; ++i)
	{
		std::cout << a[i] << ", ";
	}
	std::cout << std::endl << "----------------------" << std::endl;
}





//T should either be double or float
//there is num_elem colors in fb, and 3 x num_elem in ucfb
template <class T>
__global__ void convert_frame_buffer(const rt::RGBColor<T> * fb, uint8_t * ucfb, int num_elem)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	uint8_t r, g, b;
	rt::RGBColor<T> col = fb[i];
	r = (col.red() / (col.red()+1) * 256);
	g = (col.green() /(col.green()+1) * 256);
	b = (col.blue() / (col.blue()+1) * 256);
	ucfb[i * 3] = r;
	ucfb[i * 3+1] = g;
	ucfb[i * 3+2] = b;
}


template <class precision>
__global__ void compute_frame_buffer(rt::RGBColor<precision> * fb, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < width * height)
	{
		int i = index / height;
		int j = index % height;
		precision u = (precision)i / (precision)width;
		precision v = (precision)j / (precision)height;

		rt::RGBColor<precision> & pixel = fb[index];
		pixel.red() = u;
		pixel.green() = v;
		pixel.blue() = 0.1;
	}
}



/*
void save_image_ppm(uint8_t * buffer, size_t width, size_t height, std::string const& path)
{
	std::ofstream file(path);

	file << "P3\n";
	file << width << " " << height << "\n";
	file << "255\n";
	for (size_t j = 0; j < height; ++j)
	{
		for (size_t i = 0; i < width; ++i)
		{
			size_t index = i * height + j;
			short r = buffer[index*3];
			short g = buffer[index*3 + 1];
			short b = buffer[index*3 + 2];
			file << r << " " << g << " " << b << "\n";
		}
	}

	file.close();
}
*/

/*
void test_image()
{
	std::cout << "test image!" << std::endl;
	const size_t width = 800;
	const size_t height = 600;

	const size_t num_elem = width * height;

	//rt::RGBColorf * fbf;
	uint8_t * fbuc;

	//fbf = (rt::RGBColorf *)malloc(num_elem * sizeof(rt::RGBColorf));

	fbuc = (uint8_t *)malloc(num_elem * 3 * sizeof(uint8_t));

	rt::RGBColorf * d_fbf;
	uint8_t * d_fbuc;

	cudaMalloc((void**)&d_fbf, num_elem * sizeof(rt::RGBColorf));
	cudaMalloc((void**)&d_fbuc, num_elem * 3 * sizeof(uint8_t));
	
	const size_t num_thread = 32;
	const size_t num_block = (num_elem / num_thread)*num_thread >= num_elem ? num_elem / num_thread : num_elem / num_thread +1;
	compute_frame_buffer<<<num_block,32>>>(d_fbf, width, height);

	cudaDeviceSynchronize();

	convert_frame_buffer << <num_block, 32 >> > (d_fbf, d_fbuc, num_elem);

	cudaDeviceSynchronize();

	cudaMemcpy(fbuc, d_fbuc, num_elem * 3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);

	cudaFree(d_fbf);
	cudaFree(d_fbuc);

	save_image_ppm(fbuc, width, height, "image_test.ppm");

	free(fbuc);

}
//*/




void test_cuda_int()
{
	const int size = 100;

	int * h_a = new int[size];
	int * h_b = new int[size];
	int * h_c = new int[size];

	fill_random(h_a, size);
	fill_random(h_b, size);

	int * d_a, *d_b, *d_c;

	cudaMalloc((void**)&d_a, size * sizeof(int));
	cudaMalloc((void**)&d_b, size * sizeof(int));
	cudaMalloc((void**)&d_c, size * sizeof(int));

	cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);
	cudadd << <size, 1 >> > (d_a, d_b, d_c);

	cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

	print_array(h_a, size);
	print_array(h_b, size);
	print_array(h_c, size);

	delete[] h_a;
	delete[] h_b;
	delete[] h_c;

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	cudaDeviceReset();
}




void test_cuda_vector()
{
	const unsigned int num_elem = 256;
	const unsigned int size = num_elem * sizeof(math::Vector3f);

	math::Vector3f * h_a, *h_b, *h_c;
	math::Vector3f * d_a, *d_b, *d_c;

	h_a = (math::Vector3f*)malloc(size);
	h_b = (math::Vector3f*)malloc(size);
	h_c = (math::Vector3f*)malloc(size);

	fill_vector_array(h_a, num_elem);
	fill_vector_array(h_b, num_elem);

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	cudadd << <num_elem, 1 >> > (d_a, d_b, d_c);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	print_array(h_a, num_elem);
	print_array(h_b, num_elem);
	print_array(h_c, num_elem);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(h_a);
	free(h_b);
	free(h_c);

	cudaDeviceReset();
}
















/*

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);



__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}



template <class T>
bool my_cuda_malloc(T * & ptr, size_t size, cudaError_t && cuda_status = cudaError_t())
{
	cudaMalloc();
}



__global__ void compute_frame(float * frame_buffer_f)
{

}


int main()
{
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}
*/