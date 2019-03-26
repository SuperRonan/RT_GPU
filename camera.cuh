#pragma once


#include "cuda_runtime.h"
#include "vector.cuh"
#include "ray.cuh"
#include "constant.h"
#include <iostream>

namespace rt
{
	template <class precision=float>
	class Camera
	{
	public:

		using Vector3p = math::Vector3<precision>;

		//uint m_w, m_h;

		Vector3p m_position;

		Vector3p m_front;
		Vector3p m_right;
		Vector3p m_up;

		precision m_plane_dist;
		precision m_plane_width;
		precision m_plane_height;

		


		//Add the precomputed axis

		

	public:

		__device__ __host__ Camera(Vector3p const& position, Vector3p const& direction, Vector3p const& right, Vector3p const& up, precision plane_dist, precision plane_width, precision plane_height) :
			m_position(position),
			m_front(direction),
			m_right(right),
			m_up(up),
			m_plane_dist(plane_dist),
			m_plane_width(plane_width),
			m_plane_height(plane_height)
		{}

		/*
		__device__ __host__ Ray<precision> get_ray(uint i, uint j)const
		{
			precision u = (precision)i / (precision)m_w;
			precision v = (precision)j / (precision)m_h;
			return get_ray(u, v);
		}
		*/

		__device__ __host__ Ray<precision> get_ray(precision u, precision v)const
		{
			Vector3p ray_dir = m_front * m_plane_dist + m_right * m_plane_width * (u - 0.5) - m_up * m_plane_height * (v - 0.5);
			ray_dir.set_normalized();
			return Ray<precision>(m_position, ray_dir);
		}

		__device__ __host__ void set_direction(Vector3p const& dir)
		{
			m_front = dir;
			m_right = math::rotateZ<precision>(m_front, -HALF_PI);
			m_right[2] = 0;
			m_right.set_normalized();
			m_up = m_right ^ m_front;
		}

	};
}
