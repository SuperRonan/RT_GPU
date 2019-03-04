#pragma once


#include "cuda_runtime.h"
#include "vector.cuh"
#include "ray.cuh"
#include "raytriangleintersection.cuh"
#include "triangle.cuh"

namespace rt
{
	template <class precision=float>
	class CastedRay : public Ray<precision>
	{
	protected:
		RayTriangleIntersection<precision> m_inter;

	public:

		__device__ __host__ CastedRay(Vector3p const& p_source, Vector3p const& p_dir) :
			Ray(p_source, p_dir)
		{}

		__device__ __host__ CastedRay(Ray<precision> const& ray) :
			Ray(ray)
		{}

		__device__ __host__ void intersect(Triangle<precision> const& tri)
		{
			m_inter.update(*this, tri);
			/*
			RayTriangleIntersection<precision> new_inter(*this, tri);
			if (new_inter < m_inter)
			{
				m_inter = new_inter;
			}
			//*/
		}

		__device__ __host__ RayTriangleIntersection<precision> const& intersection()const
		{
			return m_inter;
		}
	};
}
