#pragma once


#include "vector.cuh"
#include "RGBColor.cuh"

namespace rt
{

	template <class floot>
	class DirectionalLight
	{
	protected:

		using Colorf = RGBColor<floot>;
		using Vector3f = math::Vector3<floot>;


		// position of the light (or direction for directional lights)
		Vector3f m_direction;

		Colorf m_color;

	public:

		__device__ __host__ DirectionalLight(DirectionalLight const& other) :
			m_direction(other.m_direction),
			m_color(other.m_color)
		{}

		__device__ __host__ DirectionalLight(Vector3f const& dir, Colorf const& color) :
			m_direction(dir),
			m_color(color)
		{}

		__device__ __host__ Colorf const& color()const
		{
			return m_color;
		}

		__device__ __host__ Vector3f const& direction()const
		{
			return -m_dir;
		}

		__device__ __host__ Colorf const& contribution(Vector3f const& point)const
		{
			return m_color;
		}


		__device__ __host__ Vector3f const& to_light(Vector3f const& point)const
		{
			return m_dir;
		}

		__device__ __host__ floot dist_to_light(Vector3f const& point)const
		{
			return floot(1);
		}




	};
}