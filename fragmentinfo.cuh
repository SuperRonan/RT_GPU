#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "include_directory.cuh"

namespace rt
{
	template <class floot>
	class FragIn
	{
	protected:
		using Vector3f = math::Vector3<floot>;
		
	public:

		bool facing;

		Vector3f inter_point_world;
		Vector3f source_point_world;
		Vector3f normal_world;
		Vector3f to_view_world;
		math::Vector2<floot> uv;
		//TODO Add the material
		rt::RGBColor<floot> color;

		floot z;

		math::Vector2<floot> screen_uv;
		
		


		__device__ __host__ FragIn(rt::Ray<floot> const& ray, rt::RayTriangleIntersection<floot> const& inter, math::Vector2<floot> const& screen_pos):
			facing(inter.triangle()->facing(-ray.direction())),
			inter_point_world(inter.intersection_point()),
			source_point_world(ray.source),
			normal_world(inter.triangle()->get_normal(facing)),
			to_view_world(-ray.direction()),
			uv(inter.u(), inter.v()),
			color(inter.triangle()->color()),
			z(inter.t()),
			screen_uv(screen_pos)
		{
			//facing = inter.triangle()->facing(to_view_world);
			//normal_world = inter.triangle()->get_normal(facing);
		}

	};
}