#pragma once

#include "vector.cuh"
#include "triangle.cuh"
#include "AABB.cuh"
#include <cuda_runtime_api.h>
#include <vector>


namespace rt
{
	////////////////////////////////////////////
	// A Geometry living only on the CPU Memory
	////////////////////////////////////////////
	template <class Primitive, class floot=float, class uint=unsigned int>
	class Geometry
	{
	protected:

		uint m_material_index;

		AABB<floot> m_box;

		std::vector<Primitive> * e_storage;

	public:

		Geometry(uint material_index, std::vector<Primitive> * storage):
			m_material_index(material_index),
			e_storage(storage)
		{}



		uint material_index()const
		{
			return m_material_index;
		}

		void set_material(uint i)
		{
			m_material_index = i;
		}

		void add(Primitive const& pri)
		{
			e_storage->push_back(pri);
		}

		void size()const
		{
			return e_storage->size();
		}

	};
}