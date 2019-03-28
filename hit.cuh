#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vector.cuh"
#include "RGBColor.cuh"
#include "raytriangleintersection.cuh"

namespace rt
{
	template <class floot>
	class Hit
	{
	protected:
		using Vector3f = math::Vector3<floot>;
		using Vector2f = math::Vector2<floot>;
		
	public:

		floot z;

		//TODO Add the material
		RGBColor<floot> color;

		const void * geometry;
		const void * primitive;

		Vector3f point;
		Vector3f to_view;

		bool facing;

		Vector3f normal;
		Vector3f primitive_normal;
		
		
		
		Vector2f tex_uv;
		Vector2f primitive_uv;

		Vector3f reflected;


		


		__device__ __host__ Hit(Ray<floot> const& ray, RayTriangleIntersection<floot> const& rti) :
			z(rti.t()),
			color(rti.triangle()->color()),
			geometry((const void *)rti.triangle()),
			primitive((const void *)rti.triangle()),
			point(rti.intersection_point()),
			to_view(-ray.direction()),
			facing(rti.triangle()->facing(to_view)),
			normal(rti.triangle()->get_normal(facing)),
			primitive_normal(rti.triangle()->get_normal(facing)),
			primitive_uv(Vector2f(rti.u(), rti.v())),
			tex_uv(primitive_uv),
			reflected(normal * (2 *(normal * to_view)) - to_view)
		{
			
		}

		__device__ __host__ void construct(Ray<floot> const& ray, RayTriangleIntersection<floot> const& rti)
		{
			z = (rti.t());
			color = (rti.triangle()->color());
			geometry = ((const void *)rti.triangle());
			primitive = ((const void *)rti.triangle());
			point = (rti.intersection_point());
			to_view = (-ray.direction());
			facing = (rti.triangle()->facing(to_view));
			normal = (rti.triangle()->get_normal(facing));
			primitive_normal = (rti.triangle()->get_normal(facing));
			primitive_uv = (Vector2f(rti.u(), rti.v()));
			tex_uv = (primitive_uv);
			reflected = (normal * (2 * (normal * to_view)) - to_view);
		}

	};
}