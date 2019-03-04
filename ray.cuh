#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vector.cuh"

namespace rt
{
	template <class precision=float>
	class Ray
	{
	protected:
		using Vector3p = math::Vector3<precision>;
		
	public:
		Vector3p source;

	protected:
		Vector3p m_direction;
		Vector3p m_inv;
		math::Vector3<char> m_sign;

	public:


		//I do not check if the direction is normalized
		__device__ __host__ Ray(Vector3p const& p_source, Vector3p const& p_dir) :
			source(p_source),
			m_direction(p_dir),
			m_inv(!p_dir),
			m_sign(Vector3p::make_vector(p_dir[0] < 0, p_dir[1] < 0, p_dir[2] < 0))
		{}

		//Is the copy constructor necessary?


		__device__ __host__ Vector3p const& direction()const
		{
			return m_direction;
		}

		__device__ __host__ Vector3p const& inv_direction()const
		{
			return m_inv;
		}

		__device__ __host__ math::Vector3<char> sign()const
		{
			return m_sign;
		}

		__device__ __host__ Vector3p sample_point(precision t)const
		{
			return source + t * m_direction;
		}

	};

}