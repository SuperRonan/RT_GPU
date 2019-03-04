#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "RGBColor.cuh"

namespace rt
{
	template <class T=float>
	class Material
	{
	public:

		RGBColor<T> m_emissive;
		RGBColor<T> m_diffuse;
		RGBColor<T> m_specular;


	};
}