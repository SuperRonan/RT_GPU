#pragma once

#include <cuda_runtime.h>

namespace utils
{
	template <class integer>
	inline __device__ __host__ integer divide_up(integer num, integer denum)
	{
		integer res = num / denum;
		if (res * denum < num)
		{
			res += 1;
		}
		return res;
	}

}