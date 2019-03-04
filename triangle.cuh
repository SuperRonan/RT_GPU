#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector.cuh"
#include "RGBColor.cuh"
#include <cassert>

namespace rt
{
	template <class precision=float>
	class Triangle
	{
	protected:

		using Vector3p = math::Vector3<precision>;
		using RGBColorp = RGBColor<precision>;
		using Vector2p = math::Vector2<precision>;
		//TODO use pointer
		Vector3p m_vertex[3];

		//TODO per vertex
		Vector3p m_normal;

		//TODO add material
		RGBColorp m_color;

		

		//TODO add precomputed stuff

		Vector3p m_uaxis;
		Vector3p m_vaxis;




	public:

#define TRIANGLE true
#define QUAD false

		bool is_tri = true;

		__device__ __host__ Triangle()
		{}

		__device__ __host__ Triangle(Vector3p const& a, Vector3p const& b, Vector3p const& c, RGBColorp const& color = RGBColorp(0), bool tri=true) :
			m_color(color),
			m_uaxis(b - a),
			m_vaxis(c - a),
			is_tri(tri)
		{
			m_vertex[0] = a;
			m_vertex[1] = b;
			m_vertex[2] = c;
			m_normal = m_uaxis ^ m_vaxis;
			m_normal.set_normalized();
		}

		//TODO, for now, a shallow copy if enough
		//__device__ __host__ Triangle(Triangle const& other)


		__device__ __host__ Vector3p const& get_point(uint8_t i)const
		{
			assert(i < 3);
			return m_vertex[i];
		}

		__device__ __host__ Vector3p const& get_normal()const
		{
			return m_normal;
		}

		__device__ __host__ bool facing(Vector3p const& out_dir)const
		{
			return m_normal * out_dir >= 0;
		}

		__device__ __host__ Vector3p get_normal(bool facing)const
		{
			return facing ? m_normal : -m_normal;
		}

		__device__ __host__ Vector3p const& get_origin()const
		{
			return m_vertex[0];
		}


		__device__ __host__ Vector3p sample_point(Vector2p const& uv)const
		{
			return uv[0] * m_uaxis + uv[1] * m_vaxis;
		}

		__device__ __host__ Vector3p sample_point(precision u, precision v)const
		{
			return u* m_uaxis + v * m_vaxis;
		}

		__device__ __host__ Vector3p const& u_axis()const
		{
			return m_uaxis;
		}

		__device__ __host__ Vector3p const& v_axis()const
		{
			return m_vaxis;
		}

		__device__ __host__ RGBColorp const& color()const
		{
			return m_color;
		}


	};
}