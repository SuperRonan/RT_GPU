#pragma once

#include "cuda_runtime.h"
#include "RGBColor.cuh"
#include "vector.cuh"

namespace rt
{
	template <class floot>
	class Light
	{
	protected:

		using Colorf = RGBColor<floot>;
		using Vector3f = math::Vector3<floot>;

		
		// position of the light (or direction for directional lights)
		Vector3f m_position;
		
		Colorf m_color;

	public:
		
		__device__ __host__ Light()
		{}

		__device__ __host__ Light(Light const& other):
			m_position(other.m_position),
			m_color(other.m_color)
		{}

		__device__ __host__ Light(Vector3f const& pos, Colorf const& color):
			m_position(pos),
			m_color(color)
		{}

		__device__ __host__ Colorf const& color()const
		{
			return m_color;
		}

		__device__ __host__ Colorf const& contribution(Vector3f const& point)const
		{
			floot dist2 = (point - m_position).norm2();
			floot dist = sqrt(dist2);
			return m_color / dist;
		}


		__device__ __host__ Vector3f const& to_light(Vector3f const& point)const
		{
			return m_position - point;
		}

		__device__ __host__ floot dist_to_light(Vector3f const& point)const
		{
			return to_light().norm();
		}
		

		__device__ __host__ Vector3f const& position()const
		{
			return m_position;
		}
		

	};
}