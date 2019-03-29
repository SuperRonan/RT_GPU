#pragma once

#include "vector.cuh"
#include "triangle.cuh"
#include "AABB.cuh"
#include <cuda_runtime_api.h>

namespace rt
{
	////////////////////////////////////////////
	// for the different types of geometry: use templates ? or derived classes ?
	////////////////////////////////////////////
	template <class floot=float, class uint=unsigned int>
	class Geometry
	{
	protected:

		uint m_material_index;

		AABB<floot> m_box;

		

	public:

		Geometry(uint material_index):
			m_material_index(material_index)
		{}



		uint material_index()const
		{
			return m_material_index;
		}

		void set_material(uint i)
		{
			m_material_index = i;
		}



	};
}