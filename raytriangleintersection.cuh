#pragma once

#include "cuda_runtime.h"
#include "vector.cuh"
#include "ray.cuh"
#include "triangle.cuh"

namespace rt
{
	template <class precision=float>
	class RayTriangleIntersection
	{
	protected:

		static constexpr precision epsilon(){return 1e-5;}
		
		bool m_valid;

		math::Vector3<precision> m_intersection_point;

		precision m_t;
		precision m_u;
		precision m_v;

		const Triangle<precision> * m_triangle;

		//RayTriangleIntersection(RayTriangleIntersection const& other)
		


	public:

		__device__ __host__ RayTriangleIntersection() :
			m_valid(false)
		{}

		__device__ __host__ RayTriangleIntersection(Ray<precision> const& ray, Triangle<precision> const & tri)
		{
			precision denum = tri.get_normal() * ray.direction();
			precision num = tri.get_normal() * (tri.get_origin() - ray.source);
			m_t = num / denum;
			if (m_t < epsilon())//TODO near
			{
				m_valid = false;
			}
			else
			{
				m_triangle = &tri;
				m_intersection_point = ray.sample_point(m_t);
				math::Vector3<precision> point_from_tri = m_intersection_point - tri.get_origin();
				m_u = (point_from_tri * tri.u_axis()) / tri.u_axis().norm2();
				if (m_u < 0 || m_u > 1)
				{
					m_valid = false;
				}
				else
				{
					m_v = (point_from_tri * tri.v_axis()) / tri.v_axis().norm2();
					m_valid = m_v >= 0 && tri.is_tri * m_u + m_v <= 1 + tri.is_tri * epsilon();
				}
			}
		}



		__device__ __host__ void update(Ray<precision> const& ray, Triangle<precision> const & tri)
		{
			precision denum = tri.get_normal() * ray.direction();
			precision num = tri.get_normal() * (tri.get_origin() - ray.source);
			precision next_t = num / denum;
			if(next_t > epsilon() && *this > next_t)
			{
				math::Vector3<precision> next_intersection_point = ray.sample_point(next_t);
				math::Vector3<precision> point_from_tri = next_intersection_point - tri.get_origin();
				precision next_u = (point_from_tri * tri.u_axis()) / tri.u_axis().norm2();
				if (next_u > 0 && next_u < 1 )
				{
					precision next_v = (point_from_tri * tri.v_axis()) / tri.v_axis().norm2();
					if (next_v > 0 && tri.is_tri * next_u + next_v < 1 + tri.is_tri * epsilon())
					{
						//absorb the next intersection
						m_valid = true;
						m_t = next_t;
						m_u = next_u;
						m_v = next_v;
						m_intersection_point = next_intersection_point;
						m_triangle = &tri;
					}
				}
			}
		}


		__device__ __host__ bool operator<(RayTriangleIntersection<precision> const& other)const
		{
			return (m_valid &  other.m_valid & (m_t < other.m_t)) | !other.m_valid;
		}

		__device__ __host__ bool operator>(precision t)const
		{
			return (!m_valid) || (m_t > t);
		}

		__device__ __host__ RayTriangleIntersection<precision> & operator=(RayTriangleIntersection<precision> const& other)
		{
			m_valid = other.m_valid;
			m_u = other.m_u;
			m_v = other.m_v;
			m_t = other.m_t;
			m_intersection_point = other.m_intersection_point;
			m_triangle = other.m_triangle;
		}

		__device__ __host__ bool valid()const
		{
			return m_valid;
		}

		__device__ __host__ precision u()const&
		{
			return m_u;
		}

		__device__ __host__ precision v()const&
		{
			return m_v;
		}

		__device__ __host__ precision t()const
		{
			return m_t;
		}

		__device__ __host__ const Triangle<precision> * triangle()const
		{
			return m_triangle;
		}

		__device__ __host__ math::Vector3<precision> const& intersection_point()const
		{
			return m_intersection_point;
		}
	};
}