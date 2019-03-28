#pragma once

#include "material.cuh"
#include "phong.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <iostream>

namespace rt
{
	template <class floot, class uint=unsigned int>
	class MaterialLibrary
	{
	protected:



		////////////////////////////////////////////////////
		// TODO, a custom cuda mallocator?
		// TODO or different buffers for the different types of materials? (bas idea if it means to add support for the new type of mat in the library), 
		// but it can be possible with template ... types
		////////////////////////////////////////////////////


		using Materialf = Material<floot>;


		uint d_size;
		
		//how to make this work on cuda
		Materialf ** d_materials;


		std::vector<Materialf * > m_materials;

		
		//this operation can be quite heavy
		void clean_device()
		{
			if (d_materials != nullptr)
			{
				for (uint i = 0; i < d_size; ++i)
				{
					cudaFree(d_materials[i]);
				}

				cudaFree(d_materials);
				d_materials = nullptr;
				d_size = 0;
			}
		}


	public:

		MaterialLibrary():
			d_size(0),
			d_materials(nullptr)
		{}

		~MaterialLibrary()
		{
			clean_device();
		}

		//returns the index of the material in the buffer
		uint add_material(Material * mat)
		{
			m_materials.push_back(mat);
			return m_materials.size() - 1;
		}


		void send_to_device()
		{
			clean_device();
			d_size = m_materials.size();
			cudaError_t error = cudaMalloc((void**)&d_materials, sizeof(Materialf * ) * d_size);
			if (error != cudaSuccess)
			{
				std::cerr << "Error, could not create the materail buffer!\n";
				std::cerr << error << std::endl;
				exit(-1);
			}

			//error = cudaMemcpy(d_materials, m_materials.data(), sizeof(Materialf *) * d_size);
			
			//if (error != cudaSuccess)
			//{
			//	std::cerr << "Error, could not fill the buffer material!\n";
			//	std::cerr << error << std::endl;
			//	exit(-1);
			//}

			for (uint i = 0; i < d_size; ++i)
			{
				error = cudaMalloc((void**)d_materials + i, d_materials[i]->memory_size());
				if (error != cudaSuccess)
				{
					std::cerr << "Error, could not send the material " << i << " to the device buffer\n";
					std::cerr << error << std::endl;
					
				}
				else
				{
					//of maybe a cudaMemcpy ???
					*(d_materials[i]) = m_materials[i];
				}
			}
			
		}



		void clean_all()
		{
			clean_device();
			m_materials.clear();
		}


		const Materialf ** device_buffer()const
		{
			return d_materials;
		}

		Materialf * operator[](uint i)
		{
			assert(i < m_materials.size());
			return m_materials[i];
		}

		const Materialf * operator[](uint i)const
		{
			assert(i < m_materials.size());
			return m_materials[i];
		}




	};
}