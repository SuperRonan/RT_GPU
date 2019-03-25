#pragma once
#include <cuda.h>

namespace utils
{
	template <class T, class uint=unsigned int> 
	class device_array
	{
	protected:
		uint m_capacity;
		uint m_size;
		T * d_data;
	public:
	};
}