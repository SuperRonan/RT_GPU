#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RGBColor.cuh"
#include "ray.cuh"


namespace rt
{

	template <class floot>
	class Hit;

	template <class T=float>
	class Material
	{
	protected:
		
		RGBColor<T> m_emissive;


	public:


		template <class Q>
		__device__ __host__ Material(RGBColor<Q> const& em=0):
			m_emissive(em)
		{}



		__device__ __host__ __forceinline RGBColor<T> const& get_emissive()const noexcept
		{
			return m_emissive;
		}

		__device__ __host__ __forceinline RGBColor<T> & emissive()noexcept
		{
			return m_emissive;
		}



		//__device__ __host__ virtual RGBColor<T> shader(Ray<T> const& ray, Hit<T> const& hit)

	};
}