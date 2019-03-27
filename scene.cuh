#pragma once

#include <cuda.h>
#include "triangle.cuh"
#include <thrust/device_vector.h>
#include "point_light.cuh"
#include "directional_light.cuh"
#include "RGBColor.cuh"
#include "camera.cuh"
#include "AABB.cuh"
namespace kernel
{
	
}

namespace rt
{
	template <class floot, class uint=unsigned int>
	class Scene
	{
	protected:

		uint d_world_triangles_size, d_world_triangles_capacity;
		Triangle<floot> * d_world_triangles;

		AABB<floot> m_bounding_box;

		RGBColor<floot> m_ambient = 0;

		uint d_world_lights_size, d_world_lights_capacity;
		PointLight<floot> * d_world_lights;

		Camera<floot> m_camera, * d_camera;

		uint m_diffuse_samples = 1;
		uint m_specular_samples = 1;
		uint m_light_samples = 1;

		uint max_depth = 5;





		void show_error(cudaError_t const& error)const
		{

		}


		void clean()
		{

		}



	public:



		Scene(uint triangles_default_capacity = 256, uint default_light_capacity = 16) :
			d_world_lights_size(0),
			d_world_triangles_capacity(triangles_default_capacity),
			d_world_triangles(nullptr),
			d_world_lights_size(0),
			d_world_lights_capacity(default_light_capacity),
			d_world_lights(nullptr),
			m_camera(0, { 1, 0, 0 }, { 0, 0, 1 }, { 0, 1, 0 }, floot(0.3), floot(1), floot(1)),
			d_camera(nullptr)
		{
			cudaError_t error;
			error = cudaMalloc((void**)d_world_triangles, d_world_triangles_capacity * sizeof(Triangle<floot>));
			if (error != cudaSuccess)
			{
				show_error(error);
				clean();
			}
			
			error = cudaMalloc((void**)d_world_lights, d_world_lights_capacity * sizeof(PointLight<floot>));
			if (error != cudaSuccess)
			{
				show_error(error);
				clean();
			}

			error = cudaMalloc((void**)d_camera, sizeof(Camera<floot>));
			if (error != cudaSuccess)
			{
				show_error(error);
				clean();
			}

			send_camera_to_device();
		}






		void send_camera_to_device()
		{
			assert(d_camera != nullptr);
			cudaMemcpy(d_camera, &m_camera, sizeof(Camera<floot>), cudaMemcpyHostToDevice);
		}






		void render(RGBColor<floot> * d_fb, uint width, uint height)const
		{
			
		}



	};
}