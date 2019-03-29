#pragma once

#include <cuda.h>
#include "triangle.cuh"
#include <thrust/device_vector.h>
#include "point_light.cuh"
#include "directional_light.cuh"
#include "RGBColor.cuh"
#include "camera.cuh"
#include "AABB.cuh"
#include "material_library.cuh"
#include "castedray.cuh"



namespace rt
{


	namespace kernel
	{

		inline __device__ __host__ void index_to_coords(unsigned int index, unsigned int w, unsigned int h, unsigned int & i, unsigned int & j)
		{
			i = index / w;
			j = index % w;
		}


		inline __device__ __host__ unsigned int coords_to_index(unsigned int i, unsigned int j, unsigned int w, unsigned int h)
		{
			return i * w + j;
		}


		template <class floot=float, class uint = unsigned int>
		__device__ __host__ bool intersection_full(
			Hit<floot> & res,
			const Ray<floot> & ray,
			const Triangle<floot> * triangles,
			const uint triangles_size
			)
		{
			CastedRay<floot> cray = ray;
			for (unit i = 0; i < triangles_size, ++i)
			{
				cray.intersect(triangles[i]);
			}
			bool b = false;
			if (cray.intersection().valid())
			{
				res.construct(ray, cray.intersection());
				b = true;
			}
			return b;
		}




		template <class floot=float, class uint = unsigned int>
		__device__ __host__ RGBColor<floot> send_ray(
			Ray<floot> const& ray,
			const Material<floot> ** material_library,
			const Triangle<floot> * triangles,
			const uint trianles_size,
			const PointLight<floot> * plights,
			const uint plights_size,
			const uint max_depth,
			const uint depth=0
		)
		{
			RGBColor<floot> res = 0;
			if (depth <= max_depth)
			{
				Hit<floot> hit;
				if (intersection_full(hit, ray, triangles, trianles_size))
				{
					res = hit.color;
				}
			}
			return res;
		}


		template <class floot=float, class uint=unsigned int>
		__global__ void render(
			RGBColor<floot> * frame_buffer,
			const uint width, const uint height,
			const Camera<floot> * camera,
			const Material<floot> ** material_library,
			const Triangle<floot> * triangles,
			const uint triangle_size,
			const PointLight<floot> * plights,
			const uint plights_size,
			const uint max_depth
		)
		{
			const uint i = threadIdx.x + blockIdx.x * blockDim.x;
			const uint j = threadIdx.y + blockIdx.y * blockDim.y;
			if (i < height & j < width)
			{
				const uint index = coords_to_index(i, j, width, height);
				const floot v = ((floot)i) / (floot)height;
				const floot u = floot(j) / floot(width);

				Ray<floot> ray = camera->get_ray(u, v);

				RGBColor<floot> res = send_ray(ray, material_library, triangles, triangle_size, plights, plights_size);

				frame_buffer[index] = res;
			}
		}
	}


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


		MaterialLibrary<floot, uint> m_mat_lib;




		void clean_buffers()
		{
			cudaError_t error;
			error = cudaFree(d_world_lights);
			if (error != cudaSuccess)
			{
				std::cerr << "Error, could not free the light buffer!" << std::endl;
				std::cerr << error << std::endl;
			}

			d_world_lights = nullptr;
			d_world_lights_size = 0;
		}

		void clean_all()
		{
			clean_buffers();
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
				std::cerr << "Error, Could not create the triangle buffer on the device" << std::endl;
				std::cerr << error << std::endl;
			}
			
			error = cudaMalloc((void**)d_world_lights, d_world_lights_capacity * sizeof(PointLight<floot>));
			if (error != cudaSuccess)
			{
				std::cerr << "Error, Could not create the lights buffer on the device" << std::endl;
				std::cerr << error << std::endl;
			}

			error = cudaMalloc((void**)d_camera, sizeof(Camera<floot>));
			if (error != cudaSuccess)
			{
				std::cerr << "Error, Could not create the camera buffer on the device" << std::endl;
				std::cerr << error << std::endl;
			}

			send_camera_to_device();
		}



		~Scene()
		{
			clean_all();
		}



		void send_camera_to_device()
		{
			assert(d_camera != nullptr);
			//maybe check the error
			cudaMemcpy(d_camera, &m_camera, sizeof(Camera<floot>), cudaMemcpyHostToDevice);
		}






		void render(RGBColor<floot> * d_fb, uint width, uint height)const
		{
			
		}


		const MaterialLibrary<floot, uint> & material_library()const
		{
			return m_mat_lib;
		}

		
		MaterialLibrary<floot, uint> & material_library()
		{
			return m_mat_lib;
		}


		void add_material(Material<floot> * mat)
		{
			m_mat_lib.add_material(mat);
		}
	};
}