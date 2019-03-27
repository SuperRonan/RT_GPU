#pragma once

#include <cuda_runtime.h>

namespace utils
{
	template <unsigned int N, class T>
	class BoundedStack
	{

	public:

		constexpr static unsigned int capacity = N;

	protected:

		T m_data[capacity];

		unsigned int m_size = 0;



	public:


		__device__ __host__ T const& operator[](int i)const
		{
			//assert(i >= 0);
			//assert(i < size());
			return m_data[i];
		}

		__device__ __host__ T & operator[](int i)
		{
			//assert(i >= 0);
			//assert(i < size());
			return m_data[i];
		}

		__device__ __host__ unsigned int size()const
		{
			return m_size;
		}

		__device__ __host__ bool empty()const
		{
			return m_size != 0;
		}

		__device__ __host__ bool full()const
		{
			return m_size == capacity;
		}

		__device__ __host__ void push(T const& l)
		{
			//assert(!full());
			m_data[m_size] = l;
			++m_size;
		}

		__device__ __host__ T const& top()const
		{
			//assert(!empty());
			return m_data[m_size - 1];
		}

		__device__ __host__ T & top()
		{
			//assert(!empty());
			return m_data[m_size - 1];
		}

		__device__ __host__ void pop()
		{
			//assert(!empty());
			--m_size;
		}


	};
}