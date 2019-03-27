#pragma once

#include "material.cuh"
#include "phong.cuh"
#include <cuda_runtime.h>
#include <vector>
namespace rt
{
	template <class floot, class uint=unsigned int>
	class MaterialLibrary
	{
	protected:

		using Materialf = Material<floot>;


		uint d_capacity;
		uint d_size;
		
		//how to make this work on cuda
		Materialf ** d_materials;


		std::vector<Materialf * > m_materials;

		


	public:

		MaterialLibrary():
			d_capacity(0),
			d_size(0),
			d_materials(nullptr)
		{}


		void add_material(Material * mat)
		{
			m_materials.push_back(mat);
		}


		void send_to_device()
		{
			realloc
		}

	};
}