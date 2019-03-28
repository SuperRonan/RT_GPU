#pragma once

#include "vector.cuh"
#include "triangle.cuh"
#include "AABB.cuh"

namespace rt
{
	template <class floot=float, class uint=unsigned int>
	class Geometry
	{
	protected:

		uint m_material_pointer;

		AABB<floot> m_box;



	public:

	};
}