#pragma once

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <vector>
#include <iostream>

namespace rt
{
	template <class T, class uint = unsigned int>
	class DualBuffer
	{
	protected:

		std::vector<T> m_data;
		
		
		uint d_capacity;
		uint d_size;
		T * d_data;


		void clean_device()
		{
			if (device_loaded())
			{
				cudaFree(d_data);
				d_data = nullptr;
				d_size = 0;
				d_capacity = 0;
			}
		}

		DualBuffer(DualBuffer const& other) = delete;

		DualBuffer(DualBuffer && other) = delete;

		DualBuffer& operator=(DualBuffer const& other) = delete;
		
	public:

		DualBuffer():
			d_capacity(0),
			d_size(0),
			d_data(nullptr)
		{

		}

		~DualBuffer()
		{
			clean_device();
		}

		uint size()const
		{
			return m_data.size();
		}

		uint device_size()const
		{
			return d_size;
		}

		uint device_capacity()const
		{
			return d_capacity;
		}

		const T * device_data()const
		{
			return d_data;
		}

		T * device_data()
		{
			return d_data;
		}

		const T * host_data()const
		{
			return m_data.data();
		}

		T * host_data()
		{
			return m_data.data();
		}

		bool device_loaded()const
		{
			return d_data != nullptr;
		}

		void add_element(T const& t)
		{
			m_data.push_back(t);
		}

		void send_to_device(double extra=1.0)
		{
			clean_device();
			d_data = m_data.size();
			d_capacity = std::ceil(extra * d_data);
			cudaError_t error = cudaMalloc((void**)&m_data, sizeof(T) * d_capacity);
			if (error != cudaSuccess)
			{
				std::cerr << "Error, could not create the buffer " << this << " on the device!\n";
				std::cerr << error << std::endl;
			}

			error = cudaMemcpy(d_data, m_data.data(), d_size * sizeof(T));
			if (error != cudaSuccess)
			{
				std::cerr << "Error, could not send the buffer " << this << " to the device!\n";
				std::cerr << error << std::endl;
			}

		}

	};
}