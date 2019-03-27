#pragma once

#include "bounded_stack.cuh"
#include "directional_light.cuh"
namespace rt
{
	template <class T>
	using StackN = utils::BoundedStack<10, T>;

	template <class floot>
	using LightStack = StackN<DirectionalLight<floot>>;

	template <class floot>
	using DirectionStack = StackN<math::Vector3<floot>>;
}
