#pragma once

#include <cuda.h>
#include "triangle.cuh"
#include <thrust/device_vector.h>
#include "light.cuh"
#include "RGBColor.cuh"
#include "camera.cuh"
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

		RGBColor<floot> m_ambient = 0;

		uint d_world_lights_size, d_world_lights_capacity;
		Light<floot> * d_lights;

		Camera<floot, uint> m_camera;

		uint m_diffuse_samples = 1;
		uint m_specular_samples = 1;
		uint m_light_samples = 1;

		uint max_depth = 5;


		RGBColor<floot> * d_fb;

	public:







		void render()



	};
}